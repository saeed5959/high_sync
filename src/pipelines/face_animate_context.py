# pylint: disable=R0801
"""
This module is responsible for animating faces in videos using a combination of deep learning techniques.
It provides a pipeline for generating face animations by processing video frames and extracting face features. 
The module utilizes various schedulers and utilities for efficient face animation and supports different types 
    of latents for more control over the animation process.

Functions and Classes:
- FaceAnimatePipeline: A class that extends the DiffusionPipeline class from the diffusers library to handle face animation tasks.
  - __init__: Initializes the pipeline with the necessary components (VAE, UNets, face locator, etc.).
  - prepare_latents: Generates or loads latents for the animation process, scaling them according to the scheduler's requirements.
  - prepare_extra_step_kwargs: Prepares extra keyword arguments for the scheduler step, ensuring compatibility with different schedulers.
  - decode_latents: Decodes the latents into video frames, ready for animation.

Usage:
- Import the necessary packages and classes.
- Create a FaceAnimatePipeline instance with the required components.
- Prepare the latents for the animation process.
- Use the pipeline to generate the animated video.

Note:
- This module is designed to work with the diffusers library, which provides the underlying framework for face animation using deep learning.
- The module is intended for research and development purposes, and further optimization and customization may be required for specific use cases.
"""

import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import math
import numpy as np
import torch
from diffusers import (DDIMScheduler, DiffusionPipeline,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange, repeat
from tqdm import tqdm

from src.pipelines.context import get_context_scheduler, modify_context_all
from src.models.mutual_self_attention import ReferenceAttentionControl


@dataclass
class FaceAnimatePipelineOutput(BaseOutput):
    """
    FaceAnimatePipelineOutput is a custom class that inherits from BaseOutput and represents the output of the FaceAnimatePipeline.
    
    Attributes:
        videos (Union[torch.Tensor, np.ndarray]): A tensor or numpy array containing the generated video frames.
    
    Methods:
        __init__(self, videos: Union[torch.Tensor, np.ndarray]): Initializes the FaceAnimatePipelineOutput object with the generated video frames.
    """
    videos: Union[torch.Tensor, np.ndarray]

class FaceAnimatePipeline(DiffusionPipeline):
    """
    FaceAnimatePipeline is a custom DiffusionPipeline for animating faces.
    
    It inherits from the DiffusionPipeline class and is used to animate faces by
    utilizing a variational autoencoder (VAE), a reference UNet, a denoising UNet,
    a face locator, and an image processor. The pipeline is responsible for generating
    and animating face latents, and decoding the latents to produce the final video output.
    
    Attributes:
        vae (VaeImageProcessor): Variational autoencoder for processing images.
        reference_unet (nn.Module): Reference UNet for mutual self-attention.
        denoising_unet (nn.Module): Denoising UNet for image denoising.
        face_locator (nn.Module): Face locator for detecting and cropping faces.
        image_proj (nn.Module): Image projector for processing images.
        scheduler (Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
                         EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                         DPMSolverMultistepScheduler]): Diffusion scheduler for
                         controlling the noise level.
    
    Methods:
        __init__(self, vae, reference_unet, denoising_unet, face_locator,
                 image_proj, scheduler): Initializes the FaceAnimatePipeline
                 with the given components and scheduler.
        prepare_latents(self, batch_size, num_channels_latents, width, height,
                       video_length, dtype, device, generator=None, latents=None):
                       Prepares the initial latents for video generation.
        prepare_extra_step_kwargs(self, generator, eta): Prepares extra keyword
                       arguments for the scheduler step.
        decode_latents(self, latents): Decodes the latents to produce the final
                       video output.
    """
    def __init__(
        self,
        vae,
        reference_unet,
        denoising_unet,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ]
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor: int = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True,
        )

        self.context_connection_frames = torch.zeros((20,4,3,64,64))
        self.context_connection_frames_prev = None

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_latents(
        self,
        batch_size: int,                      # Number of videos to generate in parallel
        num_channels_latents: int,           # Number of channels in the latents
        width: int,                           # Width of the video frame
        height: int,                         # Height of the video frame
        video_length: int,                   # Length of the video in frames
        dtype: torch.dtype,                 # Data type of the latents
        device: torch.device,               # Device to store the latents on
        generator: Optional[torch.Generator] = None,  # Random number generator for reproducibility
        latents: Optional[torch.Tensor] = None  # Pre-generated latents (optional)
    ):
        """
        Prepares the initial latents for video generation.

        Args:
            batch_size (int): Number of videos to generate in parallel.
            num_channels_latents (int): Number of channels in the latents.
            width (int): Width of the video frame.
            height (int): Height of the video frame.
            video_length (int): Length of the video in frames.
            dtype (torch.dtype): Data type of the latents.
            device (torch.device): Device to store the latents on.
            generator (Optional[torch.Generator]): Random number generator for reproducibility.
            latents (Optional[torch.Tensor]): Pre-generated latents (optional).

        Returns:
            latents (torch.Tensor): Tensor of shape (batch_size, num_channels_latents, width, height)
            containing the initial latents for video generation.
        """
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        """
        Prepares extra keyword arguments for the scheduler step.

        Args:
            generator (Optional[torch.Generator]): Random number generator for reproducibility.
            eta (float): The eta (η) parameter used with the DDIMScheduler. 
            It corresponds to η in the DDIM paper (https://arxiv.org/abs/2010.02502) and should be between [0, 1].

        Returns:
            dict: A dictionary containing the extra keyword arguments for the scheduler step.
        """
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def decode_latents(self, latents):
        """
        Decode the latents to produce a video.

        Parameters:
        latents (torch.Tensor): The latents to be decoded.

        Returns:
        video (torch.Tensor): The decoded video.
        video_length (int): The length of the video in frames.
        """
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(
                latents[frame_idx: frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video


    @torch.no_grad()
    def __call__(
        self,
        gt_encode,
        audio_tensor,
        connection,
        mask_main,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        motion_scale: Optional[List[torch.Tensor]] = None,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[
            int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
            audio_feature_ratio=3.0,
        )

        num_channels_latents = self.denoising_unet.in_channels

        gt_encode = gt_encode.to(device="cuda", dtype=mask_main.dtype)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            gt_encode.dtype,
            device,
            generator,
        )

        self.context_connection_frames[0] = latents[0,:,-3:]

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        gt_encode = self.ref_image_processor.preprocess(
            gt_encode, height=height, width=width)  # (bs, c, width, height)
        gt_encode = gt_encode.to(dtype=self.vae.dtype, device=self.vae.device) # f*c*h*w
        #do vae for gt image
                #do vae for gt image
        gt_encode_chunks = []
        f = int(np.ceil(gt_encode.shape[0] / 5))
        for i in range(f):
            gt_encode_chunks.append(self.vae.encode(gt_encode[i*5 : (i+1)*5]).latent_dist.mean)
        gt_encode = torch.cat(gt_encode_chunks, dim=0)
        
        gt_encode = gt_encode.unsqueeze(0) # b*c*f*h*w
        gt_encode = gt_encode * 0.18215
        gt_encode = rearrange(gt_encode, "b f c h w -> b c f h w")

#########################################
        num_warmup_steps = 0
        context_schedule="uniform"
        context_frames=12
        context_stride=1
        context_overlap=3
        context_batch_size=1

        context_scheduler = get_context_scheduler(context_schedule)

        #### old
        # context_queue = list(context_scheduler(0,num_inference_steps,latents.shape[2],context_frames,context_stride,context_overlap,))

        mask_main = mask_main.repeat(1,1,video_length,1,1)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t_i, t in enumerate(timesteps):

                if connection:
                    latents[0,:,:3] = self.context_connection_frames_prev[t_i]

                latents = latents*mask_main + gt_encode*(1-mask_main)

                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                #### new
                offset = t_i % context_frames
                offset_context_queue = list(context_scheduler(
                    0, num_inference_steps, latents.shape[2] + offset, context_frames, context_stride, context_overlap))
                
                context_queue = []
                for ctx_q in offset_context_queue:
                    new_ctx_q = [i_p - offset for i_p in ctx_q]
                    new_ctx_q = [i_p for i_p in new_ctx_q if i_p >= 0 and i_p < latents.shape[2]]
                    context_queue.append(new_ctx_q)

                #########

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                global_context = []
                for j in range(num_context_batches):
                    global_context.append(
                        context_queue[
                            j * context_batch_size : (j + 1) * context_batch_size
                        ]
                    )

                for count, context_main in enumerate(global_context):

                    context, len_context = modify_context_all(context_main)

                    mid_index = int(len(context[0])/2)

                    self.reference_unet(
                        gt_encode[:,:,context[0][mid_index]].repeat(
                            context_batch_size * (1), 1, 1, 1),
                        torch.zeros_like(t),
                        encoder_hidden_states=None,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer, do_classifier_free_guidance=do_classifier_free_guidance)

                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    c_audio_latents = torch.cat([audio_tensor[:, c] for c in context]).to(device)
                    audio_latents = torch.cat([torch.zeros_like(c_audio_latents), c_audio_latents], 0)

                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    pred = self.denoising_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=None,
                        audio_cond_fea=audio_latents if do_classifier_free_guidance else c_audio_latents,
                        return_dict=False,
                    )[0]

                    reference_control_writer.clear()
                    reference_control_reader.clear()

                    if do_classifier_free_guidance:
                        pred_uc, pred_c = pred.chunk(2)
                        pred = torch.cat([pred_uc.unsqueeze(0), pred_c.unsqueeze(0)])
                    else:
                        pred = pred.unsqueeze(1)

                    for j, c in enumerate(context_main):

                        if context_main[0][0]==0:
                            noise_pred[:, :, c[:len_context]] = noise_pred[:, :, c[:len_context]] + pred[:, j, :, :len_context]
                        else:
                            noise_pred[:, :, c[:len_context]] = noise_pred[:, :, c[:len_context]] + pred[:, j, :, -len_context:]
                            
                        counter[:, :, c[:len_context]] = counter[:, :, c[:len_context]] + 1

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if t_i == len(timesteps) - 1 or (
                    (t_i + 1) > num_warmup_steps and (t_i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if t_i < (len(timesteps) - 1):
                    self.context_connection_frames[t_i+1] = latents[0,:,-3:]

            reference_control_reader.clear()
            reference_control_writer.clear()

################################

        self.context_connection_frames_prev = self.context_connection_frames
        self.context_connection_frames = torch.zeros((num_inference_steps,4,3,64,64))

        latents = latents*mask_main + gt_encode*(1-mask_main)
        # Post-processing
        latents = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            latents = torch.from_numpy(latents)

        if not return_dict:
            return latents

        return FaceAnimatePipelineOutput(videos=latents)
