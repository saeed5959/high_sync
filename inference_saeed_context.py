import argparse
import os
import warnings
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn
import cv2
import numpy as np
from typing import List
from einops import rearrange
import librosa
import soundfile as sf
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

from src.pipelines.face_animate_context import FaceAnimatePipeline
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from image_processor import ImageProcessor
from src.utils.util import save_video, filter_non_none


warnings.filterwarnings("ignore")

class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    
    Args:
        reference_unet (UNet2DConditionModel): The UNet2DConditionModel used as a reference for inference.
        denoising_unet (UNet3DConditionModel): The UNet3DConditionModel used for denoising the input audio.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: EchoUNet3DConditionModel,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet

    def forward(self,):
        """
        empty function to override abstract function of nn Module
        """

    def get_modules(self):
        """
        Simple method to avoid too-few-public-methods pylint error
        """
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
        }


def create_mask(mask_path, mask_512_path, clip_length):
        mask = 1 - (cv2.imread(mask_path) / 255.)
        (h, w, c) = mask.shape
        new_mask = np.zeros((h, w, c+1))
        new_mask[:, :, :c] = mask
        new_mask[:, :, c] = mask[:, :, c-1]

        mask = torch.FloatTensor(new_mask).permute(2,0,1).unsqueeze(1)
        # mask = mask.repeat(1,clip_length,1,1)

        mask_512 = 1 - (cv2.imread(mask_512_path) / 255.)

        return mask, mask_512


def create_mask_alpha(w, h, coordinate):
    (left, top, right, bottom) = coordinate

    mask = np.ones((h,w,3))
    mask[top:bottom, left:right] = 0
    
    pad = min(int((right-left)/15), int((bottom-top)/15)) # 100/15=6.6% for padding
    
    for idx in range(left, right):
        mask[bottom-pad:bottom, idx] = np.array([np.linspace(0,1,pad)]*3).transpose(1,0)
        mask[top:top+pad, idx] = np.array([np.linspace(1,0,pad)]*3).transpose(1,0)
    
    for idx in range(top, bottom):
        mask[idx, right-pad:right] = np.array([np.linspace(0,1,pad)]*3).transpose(1,0)
        mask[idx, left:left+pad] = np.array([np.linspace(1,0,pad)]*3).transpose(1,0)
    
    for idx, start, end in zip(range(top, top+pad), np.linspace(1,0.7, pad), np.linspace(0.7,0, pad)):
        mask[idx, left:left+pad] = np.array([np.linspace(start,end,pad)]*3).transpose(1,0)
        mask[idx, right-pad:right] = np.array([np.linspace(end,start,pad)]*3).transpose(1,0)
    
    for idx, start, end in zip(range(bottom-pad, bottom), np.linspace(0.7,1, pad), np.linspace(0,0.7, pad)):
        mask[idx, left:left+pad] = np.array([np.linspace(start,end,pad)]*3).transpose(1,0)
        mask[idx, right-pad:right] = np.array([np.linspace(end,start,pad)]*3).transpose(1,0)
    
    return mask


def process_audio(input_audio_path, output_audio_path):  
    wav,_ = librosa.load(input_audio_path, sr=16000)
    tmp_path = output_audio_path[:-4]+"_16000.wav"
    sf.write(tmp_path, wav, 16000)

    model = pretrained.dns64().cuda()
    wav, sr = torchaudio.load(input_audio_path)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)

    with torch.no_grad():
        denoised = model(wav[None])[0]
        
    sf.write(output_audio_path, denoised.data.cpu().numpy()[0], 16000)

    return


def differ_bbox(bbox, bbox_prev, frame_w, frame_h):

    diff_x = ( np.abs(bbox[0] - bbox_prev[0]) + np.abs(bbox[2] - bbox_prev[2]) ) / frame_w
    diff_y = ( np.abs(bbox[1] - bbox_prev[1]) + np.abs(bbox[3] - bbox_prev[3]) ) / frame_h

    if ( diff_x + diff_y ) < 0.3:
        connection = True
    else:
        connection = False

    return connection
    

img_size = (512,512)
    

def inference_process(args: argparse.Namespace):
    """
    Perform inference processing.

    Args:
        args (argparse.Namespace): Command-line arguments.

    This function initializes the configuration for the inference process. It sets up the necessary
    modules and variables to prepare for the upcoming inference steps.
    """
    # 1. init config
    cli_args = filter_non_none(vars(args))
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, cli_args)
    save_path = config.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 2. runtime variables
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32

    # 3. prepare inference data
    # 3.1 prepare source image, face mask
    img_size = (config.data.source_image.width,
                config.data.source_image.height)
    clip_length = config.data.n_sample_frames
    mask_path = config.mask_path
    mask_512_path = config.mask_512_path

    temp_dir = os.path.join(save_path, "img")
    os.makedirs(temp_dir, exist_ok=True)
    os.system(f"rm -r {temp_dir}/*")

    # 3.2 prepare audio embeddings
    sample_rate = config.data.driving_audio.sample_rate
    assert sample_rate == 16000, "audio sample rate must be 16000"
    fps = config.data.export_video.fps

    ### load audio processor params
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    audio_temp_path = os.path.join(save_path, "audio.wav")
    process_audio(config.driving_audio, audio_temp_path)

    # 4. build modules
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    vae = AutoencoderKL.from_pretrained(config.vae.model_path)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.base_model_path, subfolder="unet")
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs)
    )

    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
    )

    # Freeze
    vae.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)

    # reference_unet.enable_gradient_checkpointing()
    # denoising_unet.enable_gradient_checkpointing()

    vae.eval()
    reference_unet.eval()
    denoising_unet.eval()

    net = Net(
        reference_unet,
        denoising_unet
    )


    # 5. inference
    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=net.reference_unet,
        denoising_unet=net.denoising_unet,
        scheduler=val_noise_scheduler
    )


    pipeline.to(device=device, dtype=weight_dtype)

    image_processor = ImageProcessor(img_size)

    generator = torch.manual_seed(42)

    video_stream = cv2.VideoCapture(args.source_video)
    
    whisper_feature = audio_processor.audio2feat(audio_temp_path)

    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
    # torch.save(whisper_chunks, "./cache/audio.pt")
    audio_frame_num = whisper_chunks.shape[0]
    audio_fea_final = torch.Tensor(whisper_chunks).to(dtype=vae.dtype, device=vae.device)
    # audio_fea_final = torch.tensor(torch.load("/workspace/preprocess/audio_emb_whisper/cM096XGOCG4_C1.pt")).to(dtype=vae.dtype, device=vae.device)
    audio_fea_final = audio_fea_final.unsqueeze(0)

    video_length = min(int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT)), audio_frame_num)
    if video_length < audio_frame_num:
        audio_fea_final = audio_fea_final[:, :video_length, :, :]

    times = video_length // config.inference_frames_num

    context_frames=12
    context_overlap = 3

    frames_video = []
    count_motion = 0
    count_img = 0
    count = 0
    finish = False

    bbox_prev = (0,0,0,0)

    while 1:

        still_reading, frame = video_stream.read()

        if (not still_reading) or (count_motion >= video_length):
            video_stream.release()
            finish = True
        else:
            frames_video.append(frame)
            count_motion += 1

        if len(frames_video)==0:
            break

        if ((count_motion % config.inference_frames_num) == 0) or finish:

            print(f"processing {count}/{times}")
            source_image_pixels_all, bbox_all, frame_crop_all, frame_list_all = image_processor.preprocess_frames_jump(frames_video)            

            for source_image_pixels_list, bbox_list, frame_crop_list, frame_list in zip(source_image_pixels_all, bbox_all, frame_crop_all, frame_list_all):

                print(f"generating {count}/{times}")

                if bbox_list[0]!=(0,0,0,0):
                    
                    audio_tensor = audio_fea_final[:, count_img : count_img + len(frame_list)]
                    main_length = len(frame_list)
                    diff = context_frames - main_length 

                    frame_h, frame_w, _ = frame_list[0].shape
                    connection = differ_bbox(bbox_list[0], bbox_prev, frame_w, frame_h)

                    if connection == True:
                        source_image_pixels_list = source_image_pixels_prev.extend(source_image_pixels_list)
                        audio_tensor = torch.cat((audio_tensor_prev, audio_tensor), dim=1)
                        diff -= context_overlap

                    if diff > 0:
                        source_image_pixels_list.extend([source_image_pixels_list[-1]]*diff)
                        audio_tensor = torch.cat((audio_tensor, audio_tensor[:,-1:].repeat(1,diff,1,1)), dim=1)

                    source_image_pixels = torch.cat(source_image_pixels_list, dim=0)
                    mask_main, _ = create_mask(mask_path, mask_512_path, len(frame_list))
                    mask_main = mask_main.unsqueeze(0).to(device="cuda", dtype=weight_dtype)

                    with torch.no_grad():
                        pipeline_output = pipeline(
                            gt_encode=source_image_pixels,
                            audio_tensor=audio_tensor,
                            connection=connection,
                            mask_main=mask_main,
                            width=img_size[0],
                            height=img_size[1],
                            video_length=len(source_image_pixels_list),
                            num_inference_steps=config.inference_steps,
                            guidance_scale=config.cfg_scale,
                            generator=generator
                        )

                    if connection == True:
                        images_out = pipeline_output.videos.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()[context_overlap : (main_length+context_overlap)]  # convert to [f, h, w, c]
                    else:
                        images_out = pipeline_output.videos.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()[:main_length]  # convert to [f, h, w, c]

                    images_out = np.clip(images_out * 255, 0, 255).astype(np.uint8)  # to [0, 255]

                    for image_out, bbox, frame, frame_crop in zip(images_out, bbox_list, frame_list, frame_crop_list):
                        
                        frame_h, frame_w, _ = frame.shape
                        path = os.path.join(temp_dir, f"image_{count_img}.png")
                        
                        if bbox==(0,0,0,0):
                            cv2.imwrite(path, frame)
                        else:
                            image_out = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
                            # image_out = (image_out * mask_512 + frame_crop * (1 - mask_512)).astype(np.uint8)
                            frame_in_out = frame.copy()
                            frame_in_out[bbox[1]:bbox[3], bbox[0]:bbox[2]] = cv2.resize(image_out, (bbox[2] - bbox[0], bbox[3] - bbox[1]))
                            mask = create_mask_alpha(frame_w, frame_h, bbox)
                            blurred_image = (frame * mask + frame_in_out * (1-mask)).astype(np.uint8)    
                            cv2.imwrite(path, blurred_image)
                        
                        count_img += 1

                else:
                    for frame in frame_list:
                        path = os.path.join(temp_dir, f"image_{count_img}.png")
                        cv2.imwrite(path, frame)
                        count_img += 1

                bbox_prev = bbox_list[-1]
                source_image_pixels_prev = source_image_pixels_list[-context_overlap:]
                audio_tensor_prev = audio_tensor[:,-context_overlap:]

            count += 1

            frames_video = []

        if finish:
            break

    output_file = config.output
    # save the result after all iteration
    frames_path = os.path.join(temp_dir, f"image_%d.png")
    save_video(frames_path, config.driving_audio, output_file, frame_w, frame_h)

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default="configs/inference/default.yaml")
    parser.add_argument("--source_video", type=str, required=False, help="source image")
    parser.add_argument("--driving_audio", type=str, required=False, help="driving audio")
    parser.add_argument("--output", type=str, help="output video file name", default="./cache/output.mp4")
    parser.add_argument("--inference_frames_num", type=int, help="inference frames number", default=250)


    command_line_args = parser.parse_args()

    inference_process(command_line_args)
