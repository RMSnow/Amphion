# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from tqdm import tqdm
from huggingface_hub import snapshot_download

from models.svc.vevosing.vevosing_utils import *


def vevosing_singing_style_conversion(
    raw_wav_path,
    style_ref_wav_path,
    output_path=None,
    raw_text=None,
    style_ref_text=None,
    raw_language="en",
    style_ref_language="en",
):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        task="recognition-synthesis",
        src_wav_path=raw_wav_path,
        src_text=raw_text,
        style_ref_wav_path=style_ref_wav_path,
        style_ref_wav_text=style_ref_text,
        src_text_language=raw_language,
        style_ref_wav_text_language=style_ref_language,
        timbre_ref_wav_path=raw_wav_path,  # keep the timbre as the raw wav
        use_style_tokens_as_ar_input=True,  # To use the prosody code of the raw wav
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)


def load_inference_pipeline():
    # ===== Device =====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ===== Prosody Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        local_dir="./pretrained/Vevo1.5",
        allow_patterns=["tokenizer/prosody_fvq512_6.25hz/*"],
    )
    prosody_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/prosody_fvq512_6.25hz"
    )

    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        local_dir="./pretrained/Vevo1.5",
        allow_patterns=["tokenizer/contentstyle_fvq16384_12.5hz/*"],
    )
    contentstyle_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/contentstyle_fvq16384_12.5hz"
    )

    # ===== Autoregressive Transformer =====
    model_name = "ar_emilia101k_singnet7k"

    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        local_dir="./pretrained/Vevo1.5",
        allow_patterns=[f"contentstyle_modeling/{model_name}/*"],
    )

    ar_cfg_path = f"./models/svc/vevosing/config/{model_name}.json"
    ar_ckpt_path = os.path.join(
        local_dir,
        f"contentstyle_modeling/{model_name}",
    )

    # ===== Flow Matching Transformer =====
    model_name = "fm_emilia101k_singnet7k"

    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        local_dir="./pretrained/Vevo1.5",
        allow_patterns=[f"acoustic_modeling/{model_name}/*"],
    )

    fmt_cfg_path = f"./models/svc/vevosing/config/{model_name}.json"
    fmt_ckpt_path = os.path.join(local_dir, f"acoustic_modeling/{model_name}")

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        local_dir="./pretrained/Vevo1.5",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )

    vocoder_cfg_path = "./models/svc/vevosing/config/vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # ===== Inference =====
    inference_pipeline = VevosingInferencePipeline(
        prosody_tokenizer_ckpt_path=prosody_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=contentstyle_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )
    return inference_pipeline


if __name__ == "__main__":
    inference_pipeline = load_inference_pipeline()

    output_dir = "./models/svc/vevosing/svcc2025_final/vevo1.5"
    os.makedirs(output_dir, exist_ok=True)

    ### Zero-shot Singing Style Conversion ###
    for task in ["task1", "task2"]:
        eval_file = os.path.join(
            "./models/svc/vevosing/svcc2025_final/", f"{task}.json"
        )
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        for item in tqdm(eval_data, desc=f"Processing {task}"):
            output_path = os.path.join(output_dir, item["output_path"])
            if os.path.exists(output_path):
                continue

            raw_wav_path = item["input"]["path"]
            raw_text = item["input"]["text"]
            style_ref_wav_path = item["prompt"]["path"]
            style_ref_text = item["prompt"]["text"]

            vevosing_singing_style_conversion(
                raw_wav_path=raw_wav_path,
                raw_text=raw_text,
                style_ref_wav_path=style_ref_wav_path,
                style_ref_text=style_ref_text,
                output_path=output_path,
                raw_language="en",
                style_ref_language="en",
            )
