"""
FACodec Encoder parameters: 4.21 M
FACodec Decoder parameters: 98.12 M
FACodec Redecoder parameters: 36.33 M
FACodec Total parameters: 138.66 M
"""

import sys
import os

import torch
from models.codec.ns3_codec import FACodecRedecoder
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
import librosa
import torchaudio
import os
from glob import glob
from tqdm import tqdm


def _count_parameters(model):
    model_param = 0.0
    if isinstance(model, dict):
        for key, value in model.items():
            model_param += sum(
                p.numel() for p in model[key].parameters() if p.requires_grad
            )
    else:
        model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model_param


if __name__ == "__main__":

    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    fa_encoder.eval()
    fa_decoder.eval()

    fa_redecoder = FACodecRedecoder()

    fa_redecoder.eval()

    encoder_params = _count_parameters(fa_encoder)
    decoder_params = _count_parameters(fa_decoder)
    redecoder_params = _count_parameters(fa_redecoder)
    total_params = encoder_params + decoder_params + redecoder_params
    print(f"FACodec Encoder parameters: {encoder_params/1e6:.2f} M")
    print(f"FACodec Decoder parameters: {decoder_params/1e6:.2f} M")
    print(f"FACodec Redecoder parameters: {redecoder_params/1e6:.2f} M")
    print(f"FACodec Total parameters: {total_params/1e6:.2f} M")
