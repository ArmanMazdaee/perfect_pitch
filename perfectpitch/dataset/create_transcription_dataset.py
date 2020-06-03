import os

import torch
from tqdm import tqdm

from perfectpitch.utils.data import load_audio, load_notesequence


def create_transcription_dataset(output_path, names, wav_filenames, midi_filenames):
    os.makedirs(output_path, exist_ok=True)
    for name, wav_filename, midi_filename in tqdm(
        zip(names, wav_filenames, midi_filenames),
        desc=f"creating {output_path}",
        total=len(names),
    ):
        audio = load_audio(wav_filename)
        notesequence = load_notesequence(midi_filename)
        torch.save(
            {
                "audio": audio,
                "pitches": notesequence["pitches"],
                "intervals": notesequence["intervals"],
                "velocities": notesequence["velocities"],
            },
            os.path.join(output_path, f"{name}.pt"),
        )
