import os

import numpy as np
from tqdm import tqdm

from perfectpitch.utils.audio import load_audio, audio_to_spec
from perfectpitch.utils.transcription import load_transcription


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
        np.savez(
            os.path.join(output_path, f"{name}.npz"),
            spec=spec,
            pitches=transcription["pitches"],
            intervals=transcription["intervals"],
            velocities=transcription["velocities"],
        )
