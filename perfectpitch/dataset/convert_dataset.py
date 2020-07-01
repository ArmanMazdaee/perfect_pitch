import os

import torch
from tqdm import tqdm

from perfectpitch.utils.data import load_audio, load_transcription, audio_to_spec


def convert_dataset(output_path, names, wav_filenames, midi_filenames):
    os.makedirs(output_path, exist_ok=True)
    for name, wav_filename, midi_filename in tqdm(
        zip(names, wav_filenames, midi_filenames),
        desc=f"creating {output_path}",
        total=len(names),
    ):
        audio = load_audio(wav_filename)
        spec = audio_to_spec(audio)
        transcription = load_transcription(midi_filename)
        torch.save(
            {
                "spec": spec,
                "pitches": transcription["pitches"],
                "intervals": transcription["intervals"],
                "velocities": transcription["velocities"],
            },
            os.path.join(output_path, f"{name}.pt"),
        )
