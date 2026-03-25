# Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation

[![arXiv](https://img.shields.io/badge/arXiv-2508.16332-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2508.16332)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/RMSnow/Vevo2)
[![vevo](https://img.shields.io/badge/WebPage-Demo-red.svg)](https://versasinger.github.io/)

We present **Vevo2**, a unified and controllable framework for speech and singing voice generation. Vevo2 bridges controllable speech and singing voice generation via unified prosody learning, and supports a comprehensive set of generation tasks, including:

1. Zero-shot Text-to-Speech (TTS), Text-to-Singing, and Singing Voice Synthesis (SVS)
2. Style-preserved Voice/Singing Voice Conversion (VC/SVC)
3. Style-converted Voice/Singing Voice Conversion (VC/SVC)
4. Speech/Singing Voice Editing
5. Singing Style Conversion
6. Humming-to-Singing and Instrument-to-Singing

![Vevo2](../../../imgs/svc/vevo1.5.png)

## Pre-trained Models

We have included the following pre-trained models at [🤗 RMSnow/Vevo2](https://huggingface.co/RMSnow/Vevo2):

| Model                           | Description                                                                                                                                                                                                                                                           | Pre-trained Data and Checkpoint                                                                                          |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Prosody Tokenizer**           | Converting speech/singing waveform to **coarse-grained prosody tokens** (which can also be interpreted as *melody contour* from a musical perspective). It is a single codebook VQ-VAE with a vocabulary size of 512. The frame rate is 6.25 Hz. (i.e., **56.25 bps**) | [🤗 Emilia-101k, SingNet-7k](https://huggingface.co/RMSnow/Vevo2/tree/main/tokenizer/prosody_fvq512_6.25hz)              |
| **Content-Style Tokenizer**     | Converting speech/singing waveform to **fine-grained content-style tokens**. It is a single codebook VQ-VAE with a vocabulary size of 16384. The frame rate is 12.5 Hz. (i.e., **175 bps**)                                                                           | [🤗 Emilia-101k, SingNet-7k](https://huggingface.co/RMSnow/Vevo2/tree/main/tokenizer/contentstyle_fvq16384_12.5hz)       |
| **AR Model** | A Qwen-based (Qwen2.5-0.5B) large language model post-trained to predict content-style tokens from text tokens and optionally prosody tokens, with unified prosody learning across speech and singing.                                                                               | [🤗 Emilia-101k, SingNet-7k](https://huggingface.co/RMSnow/Vevo2/tree/main/contentstyle_modeling/posttrained)            |
| **Flow-matching Transformer**   | Predicting mel-spectrogram from content-style tokens with a flow-matching transformer (350M).                                                                                                                                                                         | [🤗 Emilia-101k, SingNet-7k](https://huggingface.co/RMSnow/Vevo2/tree/main/acoustic_modeling/fm_emilia101k_singnet7k_repa) |
| **Vocoder**                     | Predicting audio from mel-spectrogram with a Vocos-based vocoder (250M).                                                                                                                                                                                             | [🤗 Emilia-101k, SingNet-7k](https://huggingface.co/RMSnow/Vevo2/tree/main/vocoder)                                      |

The training data includes:

- **Emilia-101k**: about 101k hours of speech data

- **SingNet-7k**: about 7,000 hours of internal singing voice data, preprocessed using the [SingNet pipeline](https://openreview.net/pdf?id=X6ffdf6nh3).

## Quickstart (Inference Only)

To infer with Vevo2, you need to follow the steps below:

### Clone and Environment Setup

#### 1. Clone the repository

```bash
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion
```

#### 2. Install the environment

Before start installing, making sure you are under the `Amphion` directory. If not, use `cd` to enter.

Now, we are going to install the environment. It is recommended to use conda to configure:

```bash
conda create -n vevo2 python=3.10
conda activate vevo2

pip install -r models/svc/vevo2/requirements.txt
```

### Inference Script

```sh
# FM model only (style-preserved VC/SVC)
python -m models.svc.vevo2.infer_vevo2_fm

# AR + FM (TTS, SVS, style-converted VC/SVC, Editing, Melody Control, etc.)
python -m models.svc.vevo2.infer_vevo2_ar
```

Running this will automatically download the pretrained model from HuggingFace and start the inference process. The generated audios are saved in `models/svc/vevo2/output/*.wav` by default.

## Citations

If you find this work useful for your research, please cite our paper:

```bibtex
@article{vevo2,
  title        = {Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation},
  author       = {Zhang, Xueyao and Zhang, Junan and Wang, Yuancheng and Wang, Chaoren and Chen, Yuanzhe and Jia, Dongya and Chen, Zhuo and Wu, Zhizheng},
  journal      = {{IEEE} {ACM} Trans. Audio Speech Lang. Process.},
  year         = {2026}
}
```

```bibtex
@inproceedings{vevo,
  author       = {Xueyao Zhang and Xiaohui Zhang and Kainan Peng and Zhenyu Tang and Vimal Manohar and Yingru Liu and Jeff Hwang and Dangna Li and Yuhao Wang and Julian Chan and Yuan Huang and Zhizheng Wu and Mingbo Ma},
  title        = {Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement},
  booktitle    = {{ICLR}},
  publisher    = {OpenReview.net},
  year         = {2025}
}
```

If you use the Vevo2 pre-trained models or training recipe of Amphion, please also cite:

```bibtex
@article{amphion2,
  title        = {Overview of the Amphion Toolkit (v0.2)},
  author       = {Jiaqi Li and Xueyao Zhang and Yuancheng Wang and Haorui He and Chaoren Wang and Li Wang and Huan Liao and Junyi Ao and Zeyu Xie and Yiqiao Huang and Junan Zhang and Zhizheng Wu},
  year         = {2025},
  journal      = {arXiv preprint arXiv:2501.15442},
}

@inproceedings{amphion,
    author     = {Xueyao Zhang and Liumeng Xue and Yicheng Gu and Yuancheng Wang and Jiaqi Li and Haorui He and Chaoren Wang and Ting Song and Xi Chen and Zihao Fang and Haopeng Chen and Junan Zhang and Tze Ying Tang and Lexiao Zou and Mingxuan Wang and Jun Han and Kai Chen and Haizhou Li and Zhizheng Wu},
    title      = {Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle  = {{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year       = {2024}
}
```
