# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from models.tts.maskgct.maskgct_utils import *
from huggingface_hub import hf_hub_download
import safetensors
import soundfile as sf
import os
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":

    # build model
    device = torch.device("cuda")
    cfg_path = "./models/tts/maskgct/config/maskgct.json"
    cfg = load_config(cfg_path)
    # 1. build semantic model (w2v-bert-2.0)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    # 2. build semantic codec
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    # 3. build acoustic codec
    codec_encoder, codec_decoder = build_acoustic_codec(
        cfg.model.acoustic_codec, device
    )
    # 4. build t2s model
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    # 5. build s2a model
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    local_dir = "/storage/zhangxueyao/workspace/SpeechGenerationYC/pretrained/MaskGCT-Pretrained"

    # download checkpoint
    # download semantic codec ckpt
    semantic_code_ckpt = hf_hub_download(
        "amphion/MaskGCT",
        filename="semantic_codec/model.safetensors",
        cache_dir=local_dir,
    )
    # download acoustic codec ckpt
    codec_encoder_ckpt = hf_hub_download(
        "amphion/MaskGCT",
        filename="acoustic_codec/model.safetensors",
        cache_dir=local_dir,
    )
    codec_decoder_ckpt = hf_hub_download(
        "amphion/MaskGCT",
        filename="acoustic_codec/model_1.safetensors",
        cache_dir=local_dir,
    )
    # download t2s model ckpt
    t2s_model_ckpt = hf_hub_download(
        "amphion/MaskGCT",
        filename="t2s_model/model.safetensors",
        cache_dir=local_dir,
    )
    # download s2a model ckpt
    s2a_1layer_ckpt = hf_hub_download(
        "amphion/MaskGCT",
        filename="s2a_model/s2a_model_1layer/model.safetensors",
        cache_dir=local_dir,
    )
    s2a_full_ckpt = hf_hub_download(
        "amphion/MaskGCT",
        filename="s2a_model/s2a_model_full/model.safetensors",
        cache_dir=local_dir,
    )

    # load semantic codec
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    # load acoustic codec
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
    # load t2s model
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
    # load s2a model
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )

    df = pd.read_csv(
        "/storage/zhangxueyao/workspace/SpeechGenerationYC/EvalSet/libritts-p-amc2025/libritts-p_with_ref.csv"
    )
    output_sample_ids = df["sample_id"].tolist()
    target_text_list = df["text"].tolist()
    ref_wav_path_list = df["ref_wav_path"].tolist()
    ref_text_list = df["ref_wav_text"].tolist()

    save_dir = os.path.join(
        "/storage/zhangxueyao/workspace/SpeechGenerationYC/EvalSet/libritts-p-amc2025/results/MaskGCT"
    )
    os.makedirs(save_dir, exist_ok=True)

    for i, sample_id in enumerate(tqdm(output_sample_ids)):
        ref_wav_path = os.path.join(
            "/storage/zhangxueyao/workspace/SpeechGenerationYC/EvalSet/libritts-p-amc2025/",
            ref_wav_path_list[i],
        )
        output_path = os.path.join(save_dir, f"{sample_id}.wav")

        recovered_audio = maskgct_inference_pipeline.maskgct_inference(
            ref_wav_path,
            ref_text_list[i],
            target_text_list[i],
            "en",
            "en",
            target_len=None,
        )

        sf.write(output_path, recovered_audio, 24000)
