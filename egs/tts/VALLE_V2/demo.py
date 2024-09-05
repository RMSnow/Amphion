import os
os.chdir('../../..')
print(os.getcwd()) # Ensure this is you Amphion root path, otherwise change the above path to you amphion root path
assert os.path.isfile('./README.md') # make sure the current path is Amphion root path
import sys
sys.path.append('.')
import torch

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

# prepare inference data
import librosa
import torch
wav, _ = librosa.load('./egs/tts/VALLE_V2/example.wav', sr=16000)
wav = torch.tensor(wav, dtype=torch.float32)

# The transcript of the prompt part
prompt_transcript_text = 'and keeping eternity before the eyes'

# Here are the words you want the model to output
target_transcript_text = 'It presents a unified framework that is inclusive of diverse generation tasks and models with the added bonus of being easily extendable for new applications'
from models.tts.valle_v2.g2p_processor import G2pProcessor
g2p = G2pProcessor()
prompt_transcript = g2p(prompt_transcript_text, 'en')[1]
target_transcript = g2p(target_transcript_text, 'en')[1]

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

print(f'prompt_transcript : {prompt_transcript_text}')
print(f'target_transcript : {target_transcript_text}')

import torchaudio
print("output_wav", output_wav.shape)
torchaudio.save('out.wav', output_wav.detach().cpu().squeeze(0), 16000)
print("Saved!")
