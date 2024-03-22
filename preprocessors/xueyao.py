# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import librosa
from tqdm import tqdm
import glob

from utils.util import has_existed
from utils.audio_slicer import split_utterances_from_audio


def split_into_segments(dataset_path, output_path):
    for tag in ["Read", "Sing"]:
        singer_name = "Xueyao-{}".format(tag)
        audio_wavs = glob.glob(os.path.join(dataset_path, "{}-*.wav".format(tag)))

        for wav_file in tqdm(audio_wavs):
            song_name = os.path.basename(wav_file).split(".")[0]
            output_dir = os.path.join(output_path, singer_name, song_name)

            if os.path.exists(output_dir):
                os.system("rm -rf {}".format(output_dir))
            os.makedirs(output_dir, exist_ok=True)

            split_utterances_from_audio(
                wav_file,
                output_dir,
                max_duration_of_utterance=10,
                min_interval=300,
            )


if __name__ == "__main__":
    dataset_root_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xueyaozhang/dataset/2024-Xueyao"
    split_into_segments(
        os.path.join(dataset_root_path, "wavs"),
        os.path.join(dataset_root_path, "utterances"),
    )
