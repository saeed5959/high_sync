<h1 align='center'>HIGH-SYNC : HIGH QUALITY LIP SYNC MODEL USING
DIFFUSION MODEL</h1>

<div align='center'>
    <a href='https://github.com/saeed_5959' target='_blank'>Saeed Firouzi</a><sup>1</sup>&emsp;
</div>

<br>

<div align='center'>
    <a href='https://huggingface.co/saeed-5959/high_sync'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>


## Abstraction
we propose High-Sync model as an end-to-end model for lip sync task which utilize diffusion
structure for creating high quality videos of synced with the audio.
Previous models struggle to create high quality images without artifact and if they succeed their
syncing ability was not good.
Going to create 512*512 resolution for lip sync model for the first time, has been done by this
model which can be used in real production like movie industry which their videos are in high level
of resolution.
Solving the major issue of data leakage which was stopping previous model to go to the higher level
of syncing was another success for this model.
Thanks to training the model on a very various dataset , our proposed model can work in a wild
situation like extreme head movement , bad and dark lightning and hand interrupted faces.
Through a comprehensive evaluation tests which incorporates image quality and audio syncing , we
demonstrated that our model can achieve to the high level of quality and syncing.
We have provided more visualization and test videos and also the source code and the pretraining
models which can be found at : www


## Model Structure
<img src='./imgs/model_stage2.png'>


## ⚒️ Installation

### Environment

    Ubuntu 20 or 22

### Download the Codes

```bash
  git clone https://github.com/saeed5959/high_sync
  cd high_sync
```


### Install packages with `pip`
```bash
  pip install -r requirements.txt
```

### Install ffmpeg
```bash
apt-get install ffmpeg
```

### Download pretrained weights

```shell
git lfs install
git clone https://huggingface.co/saeed-5959/high_sync pretrained_weights
```

The **pretrained_weights** is organized as follows.

```
./pretrained_weights/
├── denoising_unet-500.pth
├── reference_unet-500.pth
├── sd-vae-ft-mse
│   └── ...
├── sd-image-variations-diffusers
│   └── ...
└── audio_processor
    └── whisper_tiny.pt
```

In which **denoising_unet.pth** / **reference_unet.pth** / are the main checkpoints of **Highsync**. Other models in this hub can be also downloaded from it's original hub, thanks to their brilliant works:
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)
- [audio_processor(whisper)](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)

### Inference

1)First convert your video to fps=25

```bash
ffmpeg -i input.mp4 -r 25 out_25.mp4
```

2)Then run the python inference script:

```bash
  python -m inference --source_video "video_path.mp4" --driving_audio "audio_path.wav" --output "save_path.mp4"
```

## 🙏🏻 Acknowledgements

This work is mainly based on [EchoMimic](https://github.com/antgroup/echomimic) work.

We would like to thank the contributors to the [EchoMimic](), [AnimateDiff](https://github.com/guoyww/AnimateDiff), [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) and [MuseTalk](https://github.com/TMElyralab/MuseTalk) repositories, for their open research and exploration. 

We are also grateful to [V-Express](https://github.com/tencent-ailab/V-Express) and [hallo](https://github.com/fudan-generative-vision/hallo) for their outstanding work in the area of diffusion-based talking heads.

If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.

## 📒 Citation

If you find our work useful for your research, please consider citing the paper :

```
@misc{highsync,
  title={HIGH-SYNC : HIGH QUALITY LIP SYNC MODEL USING
DIFFUSION MODEL},
  author={Saeed Firouzi},
  year={2026},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## 🌟 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=saeed_5959/highsync&type=Date)](https://star-history.com/#saeed_5959/highsync&Date)