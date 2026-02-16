# pylint: disable=E1101,C0415,W0718,R0801
# scripts/train_stage2.py
"""
This is the main training script for stage 2 of the project. 
It imports necessary packages, defines necessary classes and functions, and trains the model using the provided configuration.

The script includes the following classes and functions:

1. Net: A PyTorch model that takes noisy latents, timesteps, reference image latents, face embeddings, 
   and face masks as input and returns the denoised latents.
2. get_attention_mask: A function that rearranges the mask tensors to the required format.
3. get_noise_scheduler: A function that creates and returns the noise schedulers for training and validation.
4. process_audio_emb: A function that processes the audio embeddings to concatenate with other tensors.
5. log_validation: A function that logs the validation information using the given VAE, image encoder, 
   network, scheduler, accelerator, width, height, and configuration.
6. train_stage2_process: A function that processes the training stage 2 using the given configuration.
7. load_config: A function that loads the configuration file from the given path.

The script also includes the necessary imports and a brief description of the purpose of the file.
"""

import argparse
import copy
import logging
import math
import os
import random
import time
import warnings
from datetime import datetime
from typing import List, Tuple

import diffusers
import mlflow
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from omegaconf import OmegaConf
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.init as init

from src.pipelines.face_animate import FaceAnimatePipeline
from src.datasets.talk_video import TalkingVideoDataset, TalkingVideoDatasetVal
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.utils.util import (compute_snr, delete_additional_ckpt,
                              import_filename, init_output_dir,
                              load_checkpoint, save_checkpoint,
                              seed_everything, tensor_to_video_new)

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

writer = SummaryWriter()

class Net(nn.Module):
    """
    The Net class defines a neural network model that combines a reference UNet2DConditionModel,
    a denoising UNet3DConditionModel, a face locator, and other components to animate a face in a static image.

    Args:
        reference_unet (UNet2DConditionModel): The reference UNet2DConditionModel used for face animation.
        denoising_unet (UNet3DConditionModel): The denoising UNet3DConditionModel used for face animation.
        face_locator (FaceLocator): The face locator model used for face animation.
        reference_control_writer: The reference control writer component.
        reference_control_reader: The reference control reader component.
        imageproj: The image projection model.
        audioproj: The audio projection model.

    Forward method:
        noisy_latents (torch.Tensor): The noisy latents tensor.
        timesteps (torch.Tensor): The timesteps tensor.
        ref_image_latents (torch.Tensor): The reference image latents tensor.
        face_emb (torch.Tensor): The face embeddings tensor.
        audio_emb (torch.Tensor): The audio embeddings tensor.
        mask (torch.Tensor): Hard face mask for face locator.
        full_mask (torch.Tensor): Pose Mask.
        face_mask (torch.Tensor): Face Mask
        lip_mask (torch.Tensor): Lip Mask
        uncond_img_fwd (bool): A flag indicating whether to perform reference image unconditional forward pass.
        uncond_audio_fwd (bool): A flag indicating whether to perform audio unconditional forward pass.

    Returns:
        torch.Tensor: The output tensor of the neural network model.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: EchoUNet3DConditionModel,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        ref_image_latents: torch.Tensor,
        audio_emb: torch.Tensor,
        uncond_img_fwd: bool = False,
        uncond_audio_fwd: bool = False,
    ):
        """
        simple docstring to prevent pylint error
        """
        audio_emb = audio_emb.to(
            device=self.reference_unet.device, dtype=self.reference_unet.dtype)

        # condition forward
        if not uncond_img_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            ref_timesteps = repeat(
                ref_timesteps,
                "b -> (repeat b)",
                repeat=ref_image_latents.size(0) // ref_timesteps.size(0),
            )
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=None,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        if uncond_audio_fwd:
            audio_emb = torch.zeros_like(audio_emb).to(
                device=audio_emb.device, dtype=audio_emb.dtype
            )

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=None,
            audio_cond_fea=audio_emb,
        ).sample

        return model_pred


def get_noise_scheduler(cfg: argparse.Namespace) -> Tuple[DDIMScheduler, DDIMScheduler]:
    """
    Create noise scheduler for training.

    Args:
        cfg (argparse.Namespace): Configuration object.

    Returns:
        Tuple[DDIMScheduler, DDIMScheduler]: Train noise scheduler and validation noise scheduler.
    """

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    return train_noise_scheduler, val_noise_scheduler


def log_validation(
    accelerator: Accelerator,
    vae: AutoencoderKL,
    net: Net,
    scheduler: DDIMScheduler,
    width: int,
    height: int,
    clip_length: int = 14,
    generator: torch.Generator = None,
    cfg: dict = None,
    save_dir: str = None,
    global_step: int = 0,
    times: int = None,
    val_dataset = None,
    weight_dtype = None
) -> None:
    """
    Log validation video during the training process.

    Args:
        accelerator (Accelerator): The accelerator for distributed training.
        vae (AutoencoderKL): The autoencoder model.
        net (Net): The main neural network model.
        scheduler (DDIMScheduler): The scheduler for noise.
        width (int): The width of the input images.
        height (int): The height of the input images.
        clip_length (int): The length of the video clips. Defaults to 24.
        generator (torch.Generator): The random number generator. Defaults to None.
        cfg (dict): The configuration dictionary. Defaults to None.
        save_dir (str): The directory to save validation results. Defaults to None.
        global_step (int): The current global step in training. Defaults to 0.
        times (int): The number of inference times. Defaults to None.
        face_analysis_model_path (str): The path to the face analysis model. Defaults to "".

    Returns:
        torch.Tensor: The tensor result of the validation.
    """
    # ori_net = accelerator.unwrap_model(net)
    # reference_unet = ori_net.reference_unet
    # denoising_unet = ori_net.denoising_unet

    generator = torch.manual_seed(42)
    # tmp_denoising_unet = copy.deepcopy(denoising_unet)
    # k = 0
    # for name, param in accelerator.unwrap_model(net).denoising_unet.named_parameters():
    #     if "motion" in name:
    #         k += 1
    #         print(f"Layer: {name}")
    #         print(f"Shape: {param.shape}")
    #         print(f"Weights: {param.data}")
        
    #     if k>3:
    #         break

    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=accelerator.unwrap_model(net).reference_unet,
        denoising_unet=accelerator.unwrap_model(net).denoising_unet,
        scheduler=scheduler,
    )
    pipeline = pipeline.to("cuda")

    dataset_len = len(val_dataset)
    sample_idx = [np.random.randint(0, dataset_len-1) for _ in range(1)]

    for idx in sample_idx:
        idx = 748
        sample = val_dataset[idx]
        same = sample["same"]
        video_path = sample["video_dir"]
        audio_path = sample["audio_path"]
        audio_emb = sample["audio_tensor"]

        target_images = sample["pixel_values_vid"].to(device="cuda", dtype=weight_dtype)
        mask = sample['mask'].unsqueeze(0).to(device="cuda", dtype=weight_dtype) # b*c*f*h*w
        
        min_time = min(audio_emb.shape[0], target_images.shape[0], 5*25)
        target_images = target_images[:min_time]
        audio_emb = audio_emb[:min_time]
        tensor_result = []
        generator = torch.manual_seed(42)

        audio_emb = audio_emb.unsqueeze(0).to(device="cuda", dtype=weight_dtype)

        try:
            pipeline_output = pipeline(
                gt_encode=target_images,
                audio_tensor=audio_emb,
                mask_main=mask,
                width=cfg.data.train_width,
                height=cfg.data.train_height,
                video_length=min_time,
                num_inference_steps=cfg.inference_steps,
                guidance_scale=cfg.cfg_scale,
                generator=generator,
            )
        except Exception as e:
            import traceback
            print("Error encountered during training step:")
            traceback.print_exc()  # Prints the full error traceback
            exit(1)

        tensor_result.append(pipeline_output.videos)

        tensor_result = torch.cat(tensor_result, dim=2)
        tensor_result = tensor_result.squeeze(0)
        tensor_result = tensor_result[:, :]
        video_name = os.path.basename(video_path).split('.')[-2]
        output_file = os.path.join(save_dir,f"{global_step}_{video_name}_{idx}_{same}.mp4")
        # save the result after all iteration
        tensor_to_video_new(tensor_result, output_file, audio_path)


    # clean up
    # del tmp_denoising_unet
    del pipeline
    torch.cuda.empty_cache()

    return tensor_result


def train_stage2_process(cfg: argparse.Namespace) -> None:
    """
    Trains the model using the given configuration (cfg).

    Args:
        cfg (dict): The configuration dictionary containing the parameters for training.

    Notes:
        - This function trains the model using the given configuration.
        - It initializes the necessary components for training, such as the pipeline, optimizer, and scheduler.
        - The training progress is logged and tracked using the accelerator.
        - The trained model is saved after the training is completed.
    """
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # create output dir for training
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    module_dir = os.path.join(save_dir, "modules")
    validation_dir = os.path.join(save_dir, "validation")
    if accelerator.is_main_process:
        init_output_dir([save_dir, checkpoint_dir, module_dir, validation_dir])

    accelerator.wait_for_everyone()

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    # Create Models
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            cfg.unet_additional_kwargs)
    ).to(device="cuda", dtype=weight_dtype)

    # load module weight from stage 1
    stage1_ckpt_dir = cfg.stage1_ckpt_dir
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, "denoising_unet-330000.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, "reference_unet-330000.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    # Freeze
    vae.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)

    # Set motion module learnable
    trainable_modules = cfg.trainable_para
    for name, module in denoising_unet.named_modules():
        if any(trainable_mod in name for trainable_mod in trainable_modules):
            for param in module.parameters():
                param.requires_grad_(True)

    # audio attention parames of denoising_unet should be updated
    # for name, param in denoising_unet.named_parameters():
    #     if ("attn2" in name) or ("norm2" in name):
    #         param.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
        audio_feature_ratio=3.0,
    )

    net = Net(
        reference_unet,
        denoising_unet,
        reference_control_writer,
        reference_control_reader,
    ).to(dtype=weight_dtype)

    # unwrap_net = accelerator.unwrap_model(net)
    # print(f"mm denoise : {net.denoising_unet.state_dict()['down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.1.to_v.weight']}")

    if cfg.resume:
        denoising_unet.load_state_dict(
            torch.load(
                cfg.resume_denoise_path,
                map_location="cpu",
            ),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(
                cfg.resume_reference_path,
                map_location="cpu",
            ),
            strict=False,
        )
        print("loaded weight from resume model")
    else:
        #load weight of net for just audio and motion
        pretrained_model = torch.load(os.path.join(cfg.echo_path, "denoising_unet.pth"),map_location="cpu",)
        weight_modules = ["motion_modules"]

        filtered_state_dict = {}
        for k, v in pretrained_model.items():
            if any(trainable_mod in k for trainable_mod in weight_modules):
                filtered_state_dict[k] = v

        m , u = net.denoising_unet.load_state_dict(filtered_state_dict, strict=False)


        # assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
        print("loaded weight from Echo")

    # get noise scheduler
    train_noise_scheduler, val_noise_scheduler = get_noise_scheduler(cfg)

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()

        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            ) from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(
        filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    # get data loader
    train_dataset = TalkingVideoDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        mask_path=cfg.data.mask_path,
        sample_rate=cfg.data.sample_rate,
        n_sample_frames=cfg.data.n_sample_frames,
        audio_margin=cfg.data.audio_margin,
        data_meta_path=cfg.data.train_path,
        wav2vec_cfg=cfg.wav2vec_config,
    )
    
    val_dataset = TalkingVideoDatasetVal(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        mask_path=cfg.data.mask_path,
        sample_rate=cfg.data.sample_rate,
        n_sample_frames=cfg.data.n_sample_frames,
        audio_margin=cfg.data.audio_margin,
        data_meta_path=cfg.data.val_path,
        wav2vec_cfg=cfg.wav2vec_config,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=16
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(
            OmegaConf.to_container(
                cfg), "config.yaml"
        )
        logger.info(f"save config to {save_dir}")
        OmegaConf.save(
            cfg, os.path.join(save_dir, "config.yaml")
        )

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 8000
    first_epoch = 0

    # # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        global_step = load_checkpoint(cfg, checkpoint_dir, accelerator)
        first_epoch = global_step // num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
                    
    for _ in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values_vid = batch["pixel_values_vid"].to(weight_dtype)

                with torch.no_grad():
                    video_length = pixel_values_vid.shape[1]
                    pixel_values_vid = rearrange(
                        pixel_values_vid, "b f c h w -> (b f) c h w"
                    )
                    latents = vae.encode(pixel_values_vid).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                uncond_img_fwd = np.random.random() < cfg.uncond_img_ratio
                uncond_audio_fwd = np.random.random() < cfg.uncond_audio_ratio

                pixel_values_ref_img = batch["pixel_values_ref_img"].to(
                    dtype=weight_dtype
                )

                ref_img_and_motion = rearrange(
                    pixel_values_ref_img, "b f c h w -> (b f) c h w"
                )

                with torch.no_grad():
                    ref_image_latents = vae.encode(
                        ref_img_and_motion
                    ).latent_dist.sample()
                    ref_image_latents = ref_image_latents * 0.18215

                mask = batch["mask"].to(device="cuda", dtype=weight_dtype)
                noise = noise * mask

                uncond_target_fwd = np.random.random() < cfg.uncond_target_ratio
                if uncond_target_fwd:
                    latents = latents * mask
                
                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                # ---- Forward!!! -----
                model_pred = net(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    ref_image_latents=ref_image_latents,
                    audio_emb=batch["audio_tensor"].to(
                        dtype=weight_dtype),
                    uncond_img_fwd=uncond_img_fwd,
                    uncond_audio_fwd=uncond_audio_fwd,
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        (model_pred*mask).float(),
                        (target*mask).float(),
                        reduction="mean",
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        (model_pred*mask).float(),
                        (target*mask).float(),
                        reduction="mean",
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    ).mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.val.validation_steps == 0:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        log_validation(
                            accelerator=accelerator,
                            vae=vae,
                            net=net,
                            scheduler=val_noise_scheduler,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            clip_length=cfg.data.n_sample_frames,
                            cfg=cfg,
                            save_dir=validation_dir,
                            global_step=global_step,
                            times=cfg.single_inference_times if cfg.single_inference_times is not None else None,
                            val_dataset=val_dataset,
                            weight_dtype=weight_dtype
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            progress_bar.set_postfix(**logs)

            writer.add_scalar("loss", loss.detach().item(), global_step)

            if (
                global_step % cfg.checkpointing_steps == 0
                or global_step == cfg.solver.max_train_steps
            ):
                # save model
                save_path = os.path.join(
                    checkpoint_dir, f"checkpoint-{global_step}")
                if accelerator.is_main_process:
                    delete_additional_ckpt(checkpoint_dir, 0)
                accelerator.wait_for_everyone()
                accelerator.save_state(save_path)

                # save model weight
                unwrap_net = accelerator.unwrap_model(net)
                if accelerator.is_main_process:
                    save_checkpoint(
                        unwrap_net.reference_unet,
                        module_dir,
                        "reference_unet",
                        global_step,
                        total_limit=1,
                    )
                    save_checkpoint(
                        unwrap_net.denoising_unet,
                        module_dir,
                        "denoising_unet",
                        global_step,
                        total_limit=1,
                    )
                    
            if global_step >= cfg.solver.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def load_config(config_path: str) -> dict:
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """

    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    if config_path.endswith(".py"):
        return import_filename(config_path).cfg
    raise ValueError("Unsupported format for config file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/train/stage2.yaml"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        train_stage2_process(config)
    except Exception as e:
        logging.error("Failed to execute the training process: %s", e)
