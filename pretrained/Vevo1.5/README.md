---
license: cc-by-nc-nd-4.0
datasets:
- amphion/Emilia-Dataset
language:
- en
- zh
- ja
- ko
- de
- fr
tags:
- tts
- vc
- svs
- svc
- music
---

# Vevo1.5

[![blog](https://img.shields.io/badge/Vevo1.5-Blog-blue.svg)](https://veiled-army-9c5.notion.site/Vevo1-5-1d2ce17b49a280b5b444d3fa2300c93a)
[![arXiv](https://img.shields.io/badge/Vevo-Paper-COLOR.svg)](https://openreview.net/pdf?id=anQDiQZhDP)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/amphion/VevoSing)

We present **Vevo1.5**, a versatile zero-shot voice imitation framework capable of modeling both speech and singing voices. This framework offers two key features: 

1. Unified speech and singing voice modeling.
2. Fine-grained control over multiple voice attributes, including text, melody, style, and melody. 

For a hands-on demonstration of Vevo1.5's capabilities, we invite readers to explore [our accompanying blog post](https://veiled-army-9c5.notion.site/Vevo1-5-1d2ce17b49a280b5b444d3fa2300c93a).

## Pre-trained Models

We have included the following pre-trained models at Amphion:

| Model                           | Description                                                                                                                                                                                                                                                           | Pre-trained Data and Checkpoint                                                                                                                                                                                                                           |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Prosody Tokenizer**           | Converting speech/singing waveform to **coarse-grained prosody tokens** (which is also be interpreted as *melody contour* from a musical perspective). It is a single codebook VQ-VAE with a vocabulary size of 512. The frame rate is 6.25 Hz. (i.e., **56.25 bps**) | [🤗 Emilia-101k, Sing-0.4k](https://huggingface.co/amphion/Vevo1.5/tree/main/tokenizer/prosody_fvq512_6.25hz)                                                                                                                                              |
| **Content-Style Tokenizer**     | Converting speech/singing waveform to **fine-grained content-style tokens**. It is a single codebook VQ-VAE with a vocabulary size of 16384. The frame rate is 12.5 Hz. (i.e., **175 bps**)                                                                           | [🤗 Emilia-101k, Sing-0.4k](https://huggingface.co/amphion/Vevo1.5/tree/main/tokenizer/contentstyle_fvq16384_12.5hz)                                                                                                                                       |
| **Auto-regressive Transformer** | Predicting content-style tokens from phone tokens (and optionally, prosody tokens) with an auto-regressive transformer (780M).                                                                                                                                        | [🤗 Emilia-101k, Sing-0.4k](https://huggingface.co/amphion/Vevo1.5/tree/main/contentstyle_modeling/ar_emilia101k_sing0.4k) <br>[🤗 Emilia-101k, SingNet-7k](https://huggingface.co/amphion/Vevo1.5/tree/main/contentstyle_modeling/ar_emilia101k_singnet7k) |
| **Flow-matching Transformer**   | Predicting mel-spectrogram from content-style tokens with a flow-matching transformer (350M).                                                                                                                                                                         | [🤗 Emilia-101k, Sing-0.4k](https://huggingface.co/amphion/Vevo1.5/tree/main/acoustic_modeling/fm_emilia101k_sing0.4k) <br> [🤗 Emilia-101k, SingNet-7k](https://huggingface.co/amphion/Vevo1.5/tree/main/acoustic_modeling/fm_emilia101k_singnet7k)        |
| **Vocoder**                     | Predicting audio from mel-spectrogram with a Vocos-based vocoder (250M).                                                                                                                                                                                              | [🤗 Emilia-101k](https://huggingface.co/amphion/Vevo/tree/main/acoustic_modeling/Vocoder) <br>[🤗 Emilia-101k, SingNet-3k](https://huggingface.co/amphion/Vevo1.5/tree/main/acoustic_modeling/Vocoder)                                                      |

The training data includes:

- **Emilia-101k**: about 101k hours of speech data
- **Sing-0.4k**: about 400 hours of open-source singing voice data as follows: 
  | Dataset Name | \#Hours   |
  | ------------ | --------- |
  | ACESinger    | 320.6     |
  | OpenSinger   | 45.7      |
  | M4Singer     | 28.4      |
  | Popbutfy     | 23.8      |
  | PopCS        | 11.5      |
  | Opencpop     | 5.1       |
  | CSD          | 3.8       |
  | **Total**    | **438.9** |
- **SingNet-7k**: about 7,000 hours of internal singing voice data, preprocessed using the [SingNet pipeline](https://openreview.net/pdf?id=X6ffdf6nh3). The SingNet-3k is a 3000-hour subset of SingNet-7k.

## Usage
You can refer to our [recipe](https://github.com/open-mmlab/Amphion/blob/vevosing/models/svc/vevosing/README.md) at GitHub for more usage details. For example, to use Vevo1.5, after you clone the Amphion github repository, you can use the script like:

```python
import os
from huggingface_hub import snapshot_download

from models.svc.vevosing.vevosing_utils import *


def vevosing_tts(
    tgt_text,
    ref_wav_path,
    ref_text=None,
    timbre_ref_wav_path=None,
    output_path=None,
    src_language="en",
    ref_language="en",
):
    if timbre_ref_wav_path is None:
        timbre_ref_wav_path = ref_wav_path

    gen_audio = inference_pipeline.inference_ar_and_fm(
        task="synthesis",
        src_wav_path=None,
        src_text=tgt_text,
        style_ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=timbre_ref_wav_path,
        style_ref_wav_text=ref_text,
        src_text_language=src_language,
        style_ref_wav_text_language=ref_language,
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)


def vevosing_editing(
    tgt_text,
    raw_wav_path,
    raw_text=None,
    output_path=None,
    raw_language="en",
    tgt_language="en",
):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        task="recognition-synthesis",
        src_wav_path=raw_wav_path,
        src_text=tgt_text,
        style_ref_wav_path=raw_wav_path,
        style_ref_wav_text=raw_text,
        src_text_language=tgt_language,
        style_ref_wav_text_language=raw_language,
        timbre_ref_wav_path=raw_wav_path,  # keep the timbre as the raw wav
        use_style_tokens_as_ar_input=True,  # To use the prosody code of the raw wav
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)


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


def vevosing_melody_control(
    tgt_text,
    tgt_melody_wav_path,
    output_path=None,
    style_ref_wav_path=None,
    style_ref_text=None,
    timbre_ref_wav_path=None,
    tgt_language="en",
    style_ref_language="en",
):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        task="recognition-synthesis",
        src_wav_path=tgt_melody_wav_path,
        src_text=tgt_text,
        style_ref_wav_path=style_ref_wav_path,
        style_ref_wav_text=style_ref_text,
        src_text_language=tgt_language,
        style_ref_wav_text_language=style_ref_language,
        timbre_ref_wav_path=timbre_ref_wav_path,
        use_style_tokens_as_ar_input=True,  # To use the prosody code
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
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=["tokenizer/prosody_fvq512_6.25hz/*"],
    )
    prosody_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/prosody_fvq512_6.25hz"
    )

    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
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
        cache_dir="./ckpts/Vevo1.5",
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
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=[f"acoustic_modeling/{model_name}/*"],
    )

    fmt_cfg_path = f"./models/svc/vevosing/config/{model_name}.json"
    fmt_ckpt_path = os.path.join(local_dir, f"acoustic_modeling/{model_name}")

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
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

    output_dir = "./models/svc/vevosing/output"
    os.makedirs(output_dir, exist_ok=True)

    ### Zero-shot Text-to-Speech and Text-to-Singing  ###
    tgt_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."
    ref_wav_path = "./models/vc/vevo/wav/arabic_male.wav"
    ref_text = "Flip stood undecided, his ears strained to catch the slightest sound."

    jaychou_path = "./models/svc/vevosing/wav/jaychou.wav"
    jaychou_text = (
        "对这个世界如果你有太多的抱怨，跌倒了就不该继续往前走，为什么，人要这么的脆弱堕"
    )
    taiyizhenren_path = "./models/svc/vevosing/wav/taiyizhenren.wav"
    taiyizhenren_text = (
        "对，这就是我，万人敬仰的太乙真人。虽然有点婴儿肥，但也掩不住我，逼人的帅气。"
    )

    # the style reference and timbre reference are same
    vevosing_tts(
        tgt_text=tgt_text,
        ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=ref_wav_path,
        output_path=os.path.join(output_dir, "zstts.wav"),
        ref_text=ref_text,
        src_language="en",
        ref_language="en",
    )

    # the style reference and timbre reference are different
    vevosing_tts(
        tgt_text=tgt_text,
        ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=jaychou_path,
        output_path=os.path.join(output_dir, "zstts_disentangled.wav"),
        ref_text=ref_text,
        src_language="en",
        ref_language="en",
    )

    # the style reference is a singing voice
    vevosing_tts(
        tgt_text="顿时，气氛变得沉郁起来。乍看之下，一切的困扰仿佛都围绕在我身边。我皱着眉头，感受着那份压力，但我知道我不能放弃，不能认输。于是，我深吸一口气，心底的声音告诉我：“无论如何，都要冷静下来，重新开始。”",
        ref_wav_path=jaychou_path,
        ref_text=jaychou_text,
        timbre_ref_wav_path=taiyizhenren_path,
        output_path=os.path.join(output_dir, "zstts_singing.wav"),
        src_language="zh",
        ref_language="zh",
    )

    ### Zero-shot Singing Editing ###
    adele_path = "./models/svc/vevosing/wav/adele.wav"
    adele_text = "Never mind, I'll find someone like you. I wish nothing but."

    vevosing_editing(
        tgt_text="Never mind, you'll find anyone like me. You wish nothing but.",
        raw_wav_path=adele_path,
        raw_text=adele_text,  # "Never mind, I'll find someone like you. I wish nothing but."
        output_path=os.path.join(output_dir, "editing_adele.wav"),
        raw_language="en",
        tgt_language="en",
    )

    vevosing_editing(
        tgt_text="对你的人生如果你有太多的期盼，跌倒了就不该低头认输，为什么啊，人要这么的彷徨堕",
        raw_wav_path=jaychou_path,
        raw_text=jaychou_text,  # "对这个世界如果你有太多的抱怨，跌倒了就不该继续往前走，为什么，人要这么的脆弱堕"
        output_path=os.path.join(output_dir, "editing_jaychou.wav"),
        raw_language="zh",
        tgt_language="zh",
    )

    ### Zero-shot Singing Style Conversion ###
    breathy_path = "./models/svc/vevosing/wav/breathy.wav"
    breathy_text = "离别没说再见你是否心酸"

    vibrato_path = "./models/svc/vevosing/wav/vibrato.wav"
    vibrato_text = "玫瑰的红，容易受伤的梦，握在手中却流失于指缝"

    vevosing_singing_style_conversion(
        raw_wav_path=breathy_path,
        raw_text=breathy_text,
        style_ref_wav_path=vibrato_path,
        style_ref_text=vibrato_text,
        output_path=os.path.join(output_dir, "ssc_breathy2vibrato.wav"),
        raw_language="zh",
        style_ref_language="zh",
    )

    ### Melody Control for Singing Synthesis ##
    humming_path = "./models/svc/vevosing/wav/humming.wav"
    piano_path = "./models/svc/vevosing/wav/piano.wav"

    # Humming to control the melody
    vevosing_melody_control(
        tgt_text="你是我的小呀小苹果，怎么爱，不嫌多",
        tgt_melody_wav_path=humming_path,
        output_path=os.path.join(output_dir, "melody_humming.wav"),
        style_ref_wav_path=taiyizhenren_path,
        style_ref_text=taiyizhenren_text,
        timbre_ref_wav_path=taiyizhenren_path,
        tgt_language="zh",
        style_ref_language="zh",
    )

    # Piano to control the melody
    vevosing_melody_control(
        tgt_text="你是我的小呀小苹果，怎么爱，不嫌多",
        tgt_melody_wav_path=piano_path,
        output_path=os.path.join(output_dir, "melody_piano.wav"),
        style_ref_wav_path=taiyizhenren_path,
        style_ref_text=taiyizhenren_text,
        timbre_ref_wav_path=taiyizhenren_path,
        tgt_language="zh",
        style_ref_language="zh",
    )

```

## Citations

If you find this work useful for your research, please cite our paper:
```bibtex
@inproceedings{vevo,
  author       = {Xueyao Zhang and Xiaohui Zhang and Kainan Peng and Zhenyu Tang and Vimal Manohar and Yingru Liu and Jeff Hwang and Dangna Li and Yuhao Wang and Julian Chan and Yuan Huang and Zhizheng Wu and Mingbo Ma},
  title        = {Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement},
  booktitle    = {{ICLR}},
  publisher    = {OpenReview.net},
  year         = {2025}
}
```

If you use the Vevo1.5 pre-trained models or training recipe of Amphion, please also cite:

```bibtex
@article{amphion2,
  title        = {Overview of the Amphion Toolkit (v0.2)},
  author       = {Jiaqi Li and Xueyao Zhang and Yuancheng Wang and Haorui He and Chaoren Wang and Li Wang and Huan Liao and Junyi Ao and Zeyu Xie and Yiqiao Huang and Junan Zhang and Zhizheng Wu},
  year         = {2025},
  journal      = {arXiv preprint arXiv:2501.15442},
}

@inproceedings{amphion,
    author={Xueyao Zhang and Liumeng Xue and Yicheng Gu and Yuancheng Wang and Jiaqi Li and Haorui He and Chaoren Wang and Ting Song and Xi Chen and Zihao Fang and Haopeng Chen and Junan Zhang and Tze Ying Tang and Lexiao Zou and Mingxuan Wang and Jun Han and Kai Chen and Haizhou Li and Zhizheng Wu},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```