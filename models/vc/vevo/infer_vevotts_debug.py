# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from huggingface_hub import snapshot_download

from models.vc.vevo.vevo_utils import *


def vevo_tts(
    src_text,
    ref_wav_path,
    timbre_ref_wav_path=None,
    output_path=None,
    ref_text=None,
    src_language="en",
    ref_language="en",
):
    if timbre_ref_wav_path is None:
        timbre_ref_wav_path = ref_wav_path

    gen_audio = inference_pipeline.inference_ar_and_fm(
        src_wav_path=None,
        src_text=src_text,
        style_ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=timbre_ref_wav_path,
        style_ref_wav_text=ref_text,
        src_text_language=src_language,
        style_ref_wav_text_language=ref_language,
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)


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

    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # ===== Autoregressive Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["contentstyle_modeling/PhoneToVq8192/*"],
    )

    ar_cfg_path = "./models/vc/vevo/config/PhoneToVq8192.json"
    ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/PhoneToVq8192")

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
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    os.makedirs("./models/vc/vevo/output_debug", exist_ok=True)

    # ===== Debug for Case1 =====
    src_text = "With tenure, Suzie'd have all the more leisure for yachting, but her publications are no good."

    ref_list = [
        {
            "wav_path": "./egs/tts/VALLE_V2/example.wav",
            "text": "and keeping eternity before the eyes though much",
        },
        {
            "wav_path": "./egs/tts/VALLE_V2/example.wav",
            "text": "and keeping eternity before the eyes, though much.",
        },
        {
            "wav_path": "./models/vc/vevo/wav/arabic_male.wav",
            "text": "Flip stood undecided, his ears strained to catch the slightest sound.",
        },
    ]

    for idx, ref in enumerate(ref_list):
        vevo_tts(
            src_text,
            ref["wav_path"],
            output_path=f"./models/vc/vevo/output_debug/case1_prompt{idx}.wav",
            ref_text=ref["text"],
            src_language="en",
            ref_language="en",
        )

    # # ===== Debug for Case2 =====
    src_text = "张飞爱吃包子，李白游览华山，奇珍异兽满山坡。"

    ref_list = [
        {
            "wav_path": "./models/vc/vevo/wav/icl_20.wav",
            "text": "对,这就是我万人敬仰的太乙真人,虽然有点婴儿肥,但也掩不住我逼人的帅气。",
        },
        {
            "wav_path": "./models/vc/vevo/wav/icl_20.wav",
            "text": "对，这就是我，万人敬仰的太乙真人。虽然有点婴儿肥，但也掩不住我，逼人的帅气。",
        },
        {
            "wav_path": "./models/vc/vevo/wav/mandarin_female.wav",
            "text": "哇，恭喜你中了大乐透，八百万可真不少呢。有什么特别的计划或想法吗？",
        },
    ]

    for idx, ref in enumerate(ref_list):
        vevo_tts(
            src_text,
            ref["wav_path"],
            output_path=f"./models/vc/vevo/output_debug/case2_prompt{idx}.wav",
            ref_text=ref["text"],
            src_language="zh",
            ref_language="zh",
        )
