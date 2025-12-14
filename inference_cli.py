#!/usr/bin/env python
"""
Tacotron2 + WaveGlow Text-to-Speech Inference Script

Usage:
    uv run python inference_cli.py --text "Hello, this is a test."
    uv run python inference_cli.py --text "Hello world" --output output.wav --sigma 0.666
"""
import argparse
import warnings
import torch
from scipy.io.wavfile import write


def load_models(device: str = "cuda"):
    """Load Tacotron2 and WaveGlow models from PyTorch Hub."""
    print("Loading Tacotron2...")
    tacotron2 = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_tacotron2',
        model_math='fp32',
        trust_repo=True
    )
    tacotron2 = tacotron2.to(device).eval()

    print("Loading WaveGlow...")
    waveglow = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_waveglow',
        model_math='fp32',
        trust_repo=True
    )
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to(device).eval()

    print("Loading text utilities...")
    utils = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_tts_utils',
        trust_repo=True
    )

    return tacotron2, waveglow, utils


def synthesize(text: str, tacotron2, waveglow, utils, device: str = "cuda", sigma: float = 0.666):
    """Synthesize speech from text."""
    # Prepare text
    sequences, lengths = utils.prepare_input_sequence([text])
    sequences = sequences.to(device)
    lengths = lengths.to(device)

    # Generate mel spectrogram
    with torch.no_grad():
        mel, _, _ = tacotron2.infer(sequences, lengths)

        # Generate audio
        audio = waveglow.infer(mel, sigma=sigma)

    # Convert to numpy
    audio = audio.squeeze().cpu().numpy()
    return audio


def main():
    parser = argparse.ArgumentParser(
        description='Text-to-Speech using Tacotron2 + WaveGlow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python inference_cli.py --text "Hello, this is a test."
    uv run python inference_cli.py --text "The quick brown fox." --output fox.wav
    uv run python inference_cli.py --text "Hello" --sigma 0.8
        """
    )
    parser.add_argument('--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output.wav', help='Output WAV file')
    parser.add_argument('--sigma', type=float, default=0.666,
                        help='WaveGlow sigma (0.666 recommended, higher=more variation)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    args = parser.parse_args()

    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"Device: {args.device}")
    print(f"Input text: {args.text}")
    print(f"Output file: {args.output}")
    print(f"Sigma: {args.sigma}")
    print()

    # Load models
    tacotron2, waveglow, utils = load_models(args.device)

    # Synthesize
    print("\nSynthesizing...")
    audio = synthesize(args.text, tacotron2, waveglow, utils, args.device, args.sigma)

    # Normalize and save
    audio = audio / max(abs(audio.max()), abs(audio.min()))  # Normalize to [-1, 1]
    audio_int16 = (audio * 32767).astype('int16')
    write(args.output, 22050, audio_int16)

    print(f"\nSaved: {args.output}")
    print(f"Duration: {len(audio) / 22050:.2f} seconds")


if __name__ == '__main__':
    main()
