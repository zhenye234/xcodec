import argparse
import os
from pathlib import Path
import sys
import torchaudio

import torch
import typing as tp
from omegaconf import OmegaConf
 
from models.soundstream_semantic import SoundStream
import torch.nn.functional as F

 
def build_codec_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model

def save_audio(wav: torch.Tensor, path: tp.Union[Path, str], sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    
    path = str(Path(path).with_suffix('.wav'))
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

 

def process_audio(input_file, output_file, rescale, args, config, soundstream):
    # Loading audio
    wav, sr = torchaudio.load(input_file)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)  # Convert to mono
    if sr != soundstream.sample_rate:
        wav = torchaudio.transforms.Resample(sr, soundstream.sample_rate)(wav)
    if config.audio_norm_scale < 1.0:
        wav = wav * config.audio_norm_scale
    
 
    wav = wav.unsqueeze(1).cuda()
    compressed = soundstream.encode(wav,   target_bw=args.bw)
    print(f"Compressed shape: {compressed.shape}")
    # Decode and save
    out = soundstream.decode(compressed)
    out = out.detach().cpu().squeeze(0)
 
    save_audio(out, output_file, 16000, rescale=rescale)
    print(f"Processed and saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='High fidelity neural audio codec for a single file.')
    parser.add_argument('--input', type=Path, default='test_audio/speech_en.flac', help='Input audio file.')
    parser.add_argument('--output', type=Path, default='test_audio_reconstruction/speech_en_nq_1.wav', help='Output audio file.')
    parser.add_argument('--resume_path', type=str, default='speech_ckpt/hubert_1k_data/xcodec_speech_hubert.pth', help='Path to model checkpoint.')
    parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping.')
    #bw 0.5-> nq 1; 1->nq 2; 2->nq 4; 4->nq 8
    parser.add_argument('-b', '--bw', type=str, default=0.5, help='Target bandwidth.')
    args = parser.parse_args()

    args.bw = float(args.bw)
    
    if not args.input.exists():
        sys.exit(f"Input file {args.input} does not exist.")

    config_path = os.path.join(os.path.dirname(args.resume_path), 'config.yaml')
    if not os.path.isfile(config_path):
        sys.exit(f"{config_path} file does not exist.")
    
    config = OmegaConf.load(config_path)
    soundstream = build_codec_model(config)
    parameter_dict = torch.load(args.resume_path)
    soundstream.load_state_dict(parameter_dict )  # Load model
    soundstream = soundstream.cuda()
    soundstream.eval()

    process_audio(args.input, args.output, args.rescale, args, config, soundstream)

if __name__ == '__main__':
    main()
