import os
os.chdir('../../..')
print(os.getcwd()) # Ensure this is you Amphion root path, otherwise change the above path to you amphion root path
assert os.path.isfile('./README.md') # make sure the current path is Amphion root path
import sys
sys.path.append('.')
import torch
import json
import librosa
import torch
import torchaudio
from tqdm import tqdm

from models.tts.valle_v2.g2p_processor import G2pProcessor
g2p = G2pProcessor()


# put your cheackpoint file (.bin) in the root path of AmphionVALLEv2
# or use your own pretrained weights
ar_model_path = 'ckpts/valle_ar_mls_196000.bin'  # huggingface-cli download amphion/valle valle_ar_mls_196000.bin valle_nar_mls_164000.bin --local-dir ckpts
nar_model_path = 'ckpts/valle_nar_mls_164000.bin'
speechtokenizer_path = 'ckpts/speechtokenizer_hubert_avg' # huggingface-cli download amphion/valle speechtokenizer_hubert_avg/SpeechTokenizer.pt speechtokenizer_hubert_avg/config.json --local-dir ckpts

# device = 'cpu' # change to 'cuda' if you have gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from models.tts.valle_v2.valle_inference import ValleInference
# change to device='cuda' to use CUDA GPU for fast inference
# change "use_vocos" to True would give better sound quality
# If you meet problem with network, you could set "use_vocos=False", though would give bad quality
model = ValleInference(ar_path=ar_model_path, nar_path=nar_model_path, speechtokenizer_path=speechtokenizer_path, device=device)
# model = ValleInference(use_vocos=False, ar_path=ar_model_path, nar_path=nar_model_path, device='cuda')

def inference(prompt_wav_path, prompt_text, input_text, output_path):
    # prepare inference data
    wav, _ = librosa.load(prompt_wav_path, sr=16000)
    wav = torch.tensor(wav, dtype=torch.float32)

    prompt_transcript = g2p(prompt_text, 'en')[1]
    target_transcript = g2p(input_text, 'en')[1]

    prompt_transcript = torch.tensor(prompt_transcript).long()
    target_transcript = torch.tensor(target_transcript).long()
    transcript = torch.cat([prompt_transcript, target_transcript], dim=-1)
    batch = {
        'speech': wav.unsqueeze(0),
        'phone_ids': transcript.unsqueeze(0),
    }

    configs = [dict(
        top_p=0.9,
        top_k=5,
        temperature=0.95,
        repeat_penalty=1.0,
        max_length=2000,
        num_beams=1,
    )] # model inference hyperparameters
    output_wav = model(batch, configs)

    torchaudio.save(output_path, output_wav.detach().cpu().squeeze(0), 16000)

def loading_inference_pairs_of_tts():
    def _load_tts_evalset(group_dir):
        json_file = os.path.join(group_dir, "evalset.json")
        wav_dir = os.path.join(group_dir, "wav")

        with open(json_file, "r") as f:
            evalset = json.load(f)

        pairs = []
        for d in evalset:
            prompt_wav_path = os.path.join(wav_dir, "{}.wav".format(d["prompt"]["uid"]))
            prompt_text = d["prompt"]["text"]
            input_text = d["input"]["text"]
            if "uid" in d["input"]:
                gt_wav_path = os.path.join(wav_dir, "{}.wav".format(d["input"]["uid"]))
            else:
                gt_wav_path = None

            output_file = d["output_path"]

            pairs.append(
                {
                    "prompt_wav_path": prompt_wav_path,
                    "input_text": input_text,
                    "gt_wav_path": gt_wav_path,
                    "output_file": output_file,
                    "prompt_text": prompt_text,
                }
            )

        return pairs

    ### g0 ###
    g0_pairs = _load_tts_evalset(os.path.join(input_root, "g0"))
    print(f"Number of pairs in g0-libritts: {len(g0_pairs)}")

    ### g1 ###
    g1_pairs = _load_tts_evalset(os.path.join(input_root, "g1"))
    print(f"Number of pairs in g1-seedeval: {len(g1_pairs)}")

    ### g2 ###
    g2_pairs = _load_tts_evalset(os.path.join(input_root, "g2"))
    print(f"Number of pairs in g2-accent: {len(g2_pairs)}")

    ### g3 ###
    g3_pairs = _load_tts_evalset(os.path.join(input_root, "g3"))
    print(f"Number of pairs in g3-emotion: {len(g3_pairs)}")

    return g0_pairs, g1_pairs, g2_pairs, g3_pairs

if __name__ == "__main__":
    input_root = "/data/home/xueyao/workspace/dataset/tts"
    output_root = "/data/home/xueyao/workspace/Amphion/results_tts"
    model_name = "valle"

    g0_pairs, g1_pairs, accent_pairs, emotion_pairs = loading_inference_pairs_of_tts()

    group_dict = {
        "g0": g0_pairs,
        "g1": g1_pairs,
        "g2": accent_pairs,
        "g3": emotion_pairs,
    }
    evalset_dict = {"g0": "LibriTTS", "g1": "SeedEval", "g2": "L2Arctic", "g3": "ESD"}

    for i in range(4):
        evalset_name = evalset_dict[f"g{i}"]
        evalset = group_dict[f"g{i}"]
        print(f"{evalset_name}...")

        output_dir = os.path.join(output_root, f"g{i}")
        os.makedirs(output_dir, exist_ok=True)

        tag = f"{model_name}-{evalset_name}"
        for sample in tqdm(evalset):
            input_text = sample["input_text"]
            prompt_wav_path = sample["prompt_wav_path"]
            prompt_text = sample["prompt_text"]
            output_file = sample["output_file"]

            filename = f"{tag}-{output_file}"
            synthesis_path = os.path.join(output_dir, filename)

            # Inference
            inference(prompt_wav_path, prompt_text, input_text, synthesis_path)