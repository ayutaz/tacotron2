# Tacotron 2 (without wavenet)

PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf).

This implementation includes **distributed** and **automatic mixed precision** support
and uses the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).

Distributed and Automatic Mixed Precision support relies on NVIDIA's [Apex] and [AMP].

Visit our [website] for audio samples using our published [Tacotron 2] and
[WaveGlow] models.

![Alignment, Predicted Mel Spectrogram, Target Mel Spectrogram](tensorboard.png)

## Architecture Documentation

For detailed architecture understanding, see:
- [CLAUDE.md](CLAUDE.md) - Overview and VITS comparison
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Encoder/Decoder/Postnet details
- [docs/ATTENTION.md](docs/ATTENTION.md) - Location Sensitive Attention
- [docs/WAVEGLOW.md](docs/WAVEGLOW.md) - WaveGlow vocoder
- [docs/TRAINING.md](docs/TRAINING.md) - Training pipeline

---

## Quick Start (Inference with Pretrained Models)

### Pre-requisites
- Python 3.10
- NVIDIA GPU + CUDA 12.x
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone repository
git clone https://github.com/ayutaz/tacotron2.git
cd tacotron2

# Install dependencies with uv
uv sync
```

### Run Inference

```bash
# Basic usage
uv run python inference_cli.py --text "Hello, this is a test."

# With custom output file
uv run python inference_cli.py --text "The quick brown fox." --output fox.wav

# Adjust sigma (variation parameter)
uv run python inference_cli.py --text "Hello world" --sigma 0.8

# CPU mode (if no GPU available)
uv run python inference_cli.py --text "Hello" --device cpu
```

The script automatically downloads pretrained Tacotron2 and WaveGlow models from NVIDIA PyTorch Hub on first run.

---

## Training Setup

### Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

### Setup
1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)
2. Clone this repo: `git clone https://github.com/ayutaz/tacotron2.git`
3. CD into this repo: `cd tacotron2`
4. Initialize submodule: `git submodule init; git submodule update`
5. Update .wav paths: `sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt`
    - Alternatively, set `load_mel_from_disk=True` in `hparams.py` and update mel-spectrogram paths
6. Install dependencies: `uv sync`

## Training
1. `python train.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download our published [Tacotron 2] model
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference demo

### CLI (Recommended)
See [Quick Start](#quick-start-inference-with-pretrained-models) section above.

### Jupyter Notebook
1. Download our published [Tacotron 2] model
2. Download our published [WaveGlow] model
3. `jupyter notebook --ip=127.0.0.1 --port=31337`
4. Load inference.ipynb

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation. 


## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time
WaveNet.

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) as described in our code.

We are inspired by [Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch)
Tacotron PyTorch implementation.

We are thankful to the Tacotron 2 paper authors, specially Jonathan Shen, Yuxuan
Wang and Zongheng Yang.


[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp