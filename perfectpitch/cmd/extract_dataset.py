import os

import librosa

from perfectpitch import constants
import perfectpitch.data.dataset
import perfectpitch.data.utils


def extract_dataset(dataset_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    dataset = perfectpitch.data.dataset.Dataset(
        dataset_path, audio=True, notesequence=True
    )
    dataset_length = len(dataset)
    for index, data in enumerate(dataset):
        print(f"extract example {index + 1} out of {dataset_length} examples")
        wav_path = os.path.join(output_path, f"{index}.wav")
        librosa.output.write_wav(wav_path, data["audio"].numpy(), constants.SAMPLE_RATE)
        midi_path = os.path.join(output_path, f"{index}.midi")
        perfectpitch.data.utils.save_notesequence(
            midi_path,
            data["notesequence"]["pitches"].numpy(),
            data["notesequence"]["intervals"].numpy(),
            data["notesequence"]["velocities"].numpy(),
        )
