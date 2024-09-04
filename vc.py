import sys
sys.path.append("/data/home/xueyao/workspace")

import torch
from Amphion.models.codec.ns3_codec import FACodecRedecoder
from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
import librosa
import torchaudio
import os
from glob import glob
from tqdm import tqdm

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

encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")
fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))

fa_encoder.eval()
fa_decoder.eval()

fa_redecoder = FACodecRedecoder()
redecoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_redecoder.bin")
fa_redecoder.load_state_dict(torch.load(redecoder_ckpt))

fa_redecoder.eval()

if torch.cuda.is_available():
    print("Using GPU...")
    fa_encoder = fa_encoder.cuda()
    fa_decoder = fa_decoder.cuda()
    fa_redecoder = fa_redecoder.cuda()


def load_wav(test_wav_path):
    test_wav = librosa.load(test_wav_path, sr=16000)[0]
    test_wav = torch.from_numpy(test_wav).float()
    test_wav = test_wav.unsqueeze(0).unsqueeze(0)

    if torch.cuda.is_available():
        test_wav = test_wav.cuda()

    return test_wav

def inference(content_wav_path, reference_wav_path, save_path):
    wav_a = load_wav(content_wav_path)
    wav_b = load_wav(reference_wav_path)

    with torch.no_grad():
        enc_out_a = fa_encoder(wav_a)
        enc_out_b = fa_encoder(wav_b)

        vq_post_emb_a, vq_id_a, _, quantized_a, spk_embs_a = fa_decoder(enc_out_a, eval_vq=False, vq=True)
        vq_post_emb_b, vq_id_b, _, quantized_b, spk_embs_b = fa_decoder(enc_out_b, eval_vq=False, vq=True)

        # convert speaker
        vq_post_emb_a_to_b = fa_redecoder.vq2emb(vq_id_a, spk_embs_b, use_residual=False)
        recon_wav_a_to_b = fa_redecoder.inference(vq_post_emb_a_to_b, spk_embs_b)

        # print("recon_wav_a_to_b", recon_wav_a_to_b.shape)
        torchaudio.save(save_path, recon_wav_a_to_b[0].cpu(), sample_rate=16000)

def get_all_wav_paths(root_dir):
    res = glob(os.path.join(root_dir, "*.wav"))
    res.sort()
    return res

def loading_inference_pairs():
    ### g0 ###
    g0_dir = os.path.join(input_root, "g0")

    g0_content_files = get_all_wav_paths(os.path.join(g0_dir, "content"))
    g0_reference_files = get_all_wav_paths(os.path.join(g0_dir, "reference"))

    g0_cont_ref_pairs = []
    for cont_path, ref_path in zip(g0_content_files, g0_reference_files):
        g0_cont_ref_pairs.append((cont_path, ref_path))

    print(f"Number of pairs in g0-vctk: {len(g0_cont_ref_pairs)}")

    ### g1 ###
    g1_dir = os.path.join(input_root, "g1")

    g1_content_files = get_all_wav_paths(os.path.join(g1_dir, "content"))
    g1_reference_files = get_all_wav_paths(os.path.join(g1_dir, "reference"))

    g1_cont_ref_pairs = []
    for cont_path, ref_path in zip(g1_content_files, g1_reference_files):
        g1_cont_ref_pairs.append((cont_path, ref_path))

    print(f"Number of pairs in g1-seedeval: {len(g1_cont_ref_pairs)}")

    ### g2: accent ###
    accent_dir = os.path.join(input_root, "g2")

    accent_content_files = get_all_wav_paths(os.path.join(accent_dir, "content"))
    accent_reference_files = get_all_wav_paths(os.path.join(accent_dir, "reference"))

    accent_cont_ref_pairs = []
    for cont_path in accent_content_files:
        for ref_path in accent_reference_files:
            accent_cont_ref_pairs.append((cont_path, ref_path))

    print(f"Number of pairs in g2-accent: {len(accent_cont_ref_pairs)}")

    ### g3: emotion ###
    emotion_dir = os.path.join(input_root, "g3")

    emotion_content_files = get_all_wav_paths(os.path.join(emotion_dir, "content"))
    emotion_reference_files = get_all_wav_paths(os.path.join(emotion_dir, "reference"))

    emotion_cont_ref_pairs = []
    for cont_path in emotion_content_files:
        for ref_path in emotion_reference_files:
            emotion_cont_ref_pairs.append((cont_path, ref_path))

    print(f"Number of pairs in g3-emotion: {len(emotion_cont_ref_pairs)}")

    return (
        g0_cont_ref_pairs,
        g1_cont_ref_pairs,
        accent_cont_ref_pairs,
        emotion_cont_ref_pairs,
    )


if __name__ == "__main__":
    input_root = "/data/home/xueyao/workspace/dataset/vc"
    output_root = "/data/home/xueyao/workspace/Amphion/results"
    model_name = "facodec"

    g0_cont_ref_pairs,g1_cont_ref_pairs,accent_cont_ref_pairs,emotion_cont_ref_pairs, = loading_inference_pairs()

    group_dict = {
        "g0": g0_cont_ref_pairs,
        "g1": g1_cont_ref_pairs,
        "g2": accent_cont_ref_pairs,
        "g3": emotion_cont_ref_pairs,
    }
    evalset_dict = {"g0": "VCTK", "g1": "SeedEval", "g2": "L2Arctic", "g3": "ESD"}

    for group, content_reference_paths_pairs in group_dict.items():
        evalset_name = evalset_dict[group]
        output_dir = os.path.join(output_root, group)
        os.makedirs(output_dir, exist_ok=True)

        for cont_path, ref_path in tqdm(content_reference_paths_pairs):
            tag = f"{model_name}-{evalset_name}"
            cont_name = os.path.basename(cont_path).split(".")[0]
            ref_name = os.path.basename(ref_path).split(".")[0]
            filename = f"{tag}-{cont_name}-{ref_name}.wav"
            conversion_path = os.path.join(output_dir, filename)

            # Inference
            inference(cont_path, ref_path, conversion_path)
