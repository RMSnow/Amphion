# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from tqdm import tqdm
import argparse
from huggingface_hub import snapshot_download

from models.vc.vevo.vevo_utils import *


def vevo_timbre(content_wav_path, reference_wav_path, output_path):
    gen_audio = inference_pipeline.inference_fm(
        src_wav_path=content_wav_path,
        timbre_ref_wav_path=reference_wav_path,
        flow_matching_steps=32,
    )
    save_audio(gen_audio, output_path=output_path)


def load_evalset(evalset_name):
    evalset_file = os.path.join(evalset_root, evalset_name, "evalset.json")
    with open(evalset_file, "r") as f:
        evalset = json.load(f)
    return evalset


if __name__ == "__main__":
    # ===== Device =====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )
    tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # ===== Flow Matching Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )

    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )

    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # ===== Inference =====
    inference_pipeline = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=tokenizer_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--evalset_root", type=str, required=True)
    parser.add_argument("--eval_setting", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True, help="tts, vc, or svc")
    args = parser.parse_args()

    evalset_root = args.evalset_root
    evalset_name = args.eval_setting
    evalset = load_evalset(evalset_name)

    print("\nFor {}...".format(evalset_name))
    save_dir = os.path.join(args.save_root, args.task_name, evalset_name)
    os.makedirs(save_dir, exist_ok=True)

    for item in tqdm(evalset):
        src_wav_path = os.path.join(
            evalset_root, evalset_name, "wav", "{}.wav".format(item["input"]["uid"])
        )
        ref_wav_path = os.path.join(
            evalset_root,
            evalset_name,
            "wav",
            "{}.wav".format(item["prompt"]["uid"]),
        )
        output_filename = "{}-{}-{}".format(
            args.model_name, evalset_name, item["output_path"]
        )

        output_path = os.path.join(save_dir, output_filename)
        if os.path.exists(output_path):
            continue

        vevo_timbre(src_wav_path, ref_wav_path, output_path)
