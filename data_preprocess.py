# pylint: disable=W1203,W0718
"""
This module is used to process videos to prepare data for training. It utilizes various libraries and models
to perform tasks such as video frame extraction, audio extraction, face mask generation, and face embedding extraction.
The script takes in command-line arguments to specify the input and output directories, GPU status, level of parallelism,
and rank for distributed processing.

Usage:
    python -m scripts.data_preprocess --input_dir /path/to/video_dir --dataset_name dataset_name --gpu_status --parallelism 4 --rank 0

Example:
    python -m scripts.data_preprocess -i data/videos -o data/output -g -p 4 -r 0
"""
import argparse
import logging
import os
from pathlib import Path
from typing import List
import soundfile as sf
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import torch
from tqdm import tqdm

from src.models.whisper.audio2feature import Audio2Feature
from src.datasets.image_processor import ImageProcessorForDataProcessing
from src.utils.util import convert_video_to_images, extract_audio_from_videos

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# model = pretrained.dns64().cuda()

def process_audio(input_audio_path):  
    
    wav, sr = torchaudio.load(input_audio_path)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)

    with torch.no_grad():
        denoised = model(wav[None])[0]
        
    sf.write(input_audio_path, denoised.data.cpu().numpy()[0], 16000)

    return


def process_single_video(video_path: Path,
                         output_dir: Path,
                         audio_processor: Audio2Feature,
                         step: int) -> None:
    """
    Process a single video file.

    Args:
        video_path (Path): Path to the video file.
        output_dir (Path): Directory to save the output.
        audio_processor (AudioProcessor): Audio processor object.
        gpu_status (bool): Whether to use GPU for processing.
    """
    assert video_path.exists(), f"Video path {video_path} does not exist"

    audio_emb_dir = output_dir / "audio_emb_whisper"
    audio_emb_dir.mkdir(parents=True, exist_ok=True)

    # try:
    if step == 1:
        images_output_dir = output_dir / 'images' / video_path.stem
        images_output_dir.mkdir(parents=True, exist_ok=True)
        images_output_dir = convert_video_to_images(
            video_path, images_output_dir)
        logging.info(f"Images saved to: {images_output_dir}")

        audio_output_dir = output_dir / 'audios'
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        audio_output_path = audio_output_dir / f'{video_path.stem}.wav'
        audio_output_path = extract_audio_from_videos(
            video_path, audio_output_path)
        logging.info(f"Audio extracted to: {audio_output_path}")

    else:
        audio_path = output_dir / "audios" / f"{video_path.stem}.wav"
        # process_audio(str(audio_path))
        audio_emb = audio_processor.audio2feat(str(audio_path))
        audio_emb = audio_processor.feature2chunks(feature_array=audio_emb, fps=25)
        torch.save(audio_emb, str(
            audio_emb_dir / f"{video_path.stem}.pt"))
    # except Exception as e:
    #     logging.error(f"Failed to process video {video_path}: {e}")


def process_all_videos(input_video_list: List[Path], output_dir: Path, step: int) -> None:
    """
    Process all videos in the input list.

    Args:
        input_video_list (List[Path]): List of video paths to process.
        output_dir (Path): Directory to save the output.
        gpu_status (bool): Whether to use GPU for processing.
    """
    whisper_model_path = "./pretrained_weights/audio_processor/whisper_tiny.pt"

    audio_processor = Audio2Feature(model_path=whisper_model_path, device="cuda") if step==2 else None

    for video_path in tqdm(input_video_list, desc="Processing videos"):
        process_single_video(video_path, output_dir, audio_processor, step)


def get_video_paths(source_dir: Path, parallelism: int, rank: int) -> List[Path]:
    """
    Get paths of videos to process, partitioned for parallel processing.

    Args:
        source_dir (Path): Source directory containing videos.
        parallelism (int): Level of parallelism.
        rank (int): Rank for distributed processing.

    Returns:
        List[Path]: List of video paths to process.
    """
    video_paths = [item for item in sorted(
        source_dir.iterdir()) if item.is_file() and item.suffix == '.mp4']
    return [video_paths[i] for i in range(len(video_paths)) if i % parallelism == rank]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process videos to prepare data for training. Run this script twice with different GPU status parameters."
    )
    parser.add_argument("-i", "--input_dir", type=Path,
                        required=True, help="Directory containing videos")
    parser.add_argument("-o", "--output_dir", type=Path,
                        help="Directory to save results, default is parent dir of input dir")
    parser.add_argument("-s", "--step", type=int, default=1,
                        help="Specify data processing step 1 or 2, you should run 1 and 2 sequently")
    parser.add_argument("-p", "--parallelism", default=1,
                        type=int, help="Level of parallelism")
    parser.add_argument("-r", "--rank", default=0, type=int,
                        help="Rank for distributed processing")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir.parent

    video_path_list = get_video_paths(
        args.input_dir, args.parallelism, args.rank)

    videos_out_path = os.path.join(args.output_dir, "videos")
    # os.system(f"cp -r {args.input_dir} {videos_out_path}")
    
    if not video_path_list:
        logging.warning("No videos to process.")
    else:
        process_all_videos(video_path_list, args.output_dir, args.step)
