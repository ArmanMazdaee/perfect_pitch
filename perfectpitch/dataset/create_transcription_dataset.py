import os
import multiprocessing

import numpy as np
from tqdm import tqdm

from perfectpitch.utils.data import load_spec, load_notesequence


def _save_sample(args):
    wav_filename, midi_filename, sample_filename = args
    spec = load_spec(wav_filename)
    notesequence = load_notesequence(midi_filename)
    np.savez(
        sample_filename,
        spec=spec,
        notesequence_pitches=notesequence["pitches"],
        notesequence_intervals=notesequence["intervals"],
        notesequence_velocities=notesequence["velocities"],
    )


def create_transcription_dataset(output_path, names, wav_filenames, midi_filenames):
    os.makedirs(output_path, exist_ok=True)
    sample_filenames = [os.path.join(output_path, f"{name}.npz") for name in names]
    args = zip(wav_filenames, midi_filenames, sample_filenames)
    with multiprocessing.Pool() as pool:
        results = pool.imap_unordered(_save_sample, args)
        for _ in tqdm(results, desc=f"creating {output_path}", total=len(names)):
            pass
