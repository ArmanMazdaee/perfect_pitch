import os
import multiprocessing

import numpy as np
from tqdm import tqdm

from perfectpitch.utils.audio import load_audio, audio_to_spec
from perfectpitch.utils.transcription import load_transcription


def _convert_sample(args):
    sample_filenames, wav_filename, midi_filename = args
    audio = load_audio(wav_filename)
    spec = audio_to_spec(audio)
    transcription = load_transcription(midi_filename)
    np.savez(
        sample_filenames,
        spec=spec,
        pitches=transcription["pitches"],
        intervals=transcription["intervals"],
        velocities=transcription["velocities"],
    )


def convert_dataset(output_path, names, wav_filenames, midi_filenames):
    os.makedirs(output_path, exist_ok=True)
    sample_filenames = [os.path.join(output_path, f"{name}.npz") for name in names]
    args = zip(sample_filenames, wav_filenames, midi_filenames)
    with multiprocessing.Pool() as pool:
        results = pool.imap_unordered(_convert_sample, args)
        for _ in tqdm(results, total=len(names), desc=f"creating {output_path}",):
            pass
