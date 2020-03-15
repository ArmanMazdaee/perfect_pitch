import os
import re
from glob import glob

import numpy as np
import librosa
from tqdm import tqdm

from perfectpitch import constants
from perfectpitch.utils.data import (
    audio_to_spec,
    load_notesequence,
    notesequence_to_pianoroll,
)


def _id_from_filename(filename):
    return re.match(r".*MAPS_MUS-(.*)_[^_]+\.[^.]+", filename).group(1)


def _set_length(tensor, length):
    if tensor.shape[1] > length:
        return tensor[:, :length]
    elif tensor.shape[1] < length:
        return np.pad(tensor, [(0, 0), (0, length - tensor.shape[1])])
    return tensor


def _create_samples(spec, pianoroll, min_length, max_length):
    if min_length is None:
        min_length = 0
    if max_length is None:
        max_length = spec.shape[1]

    for start_index in range(0, spec.shape[1], max_length):
        end_index = min(spec.shape[1], start_index + max_length)
        if end_index - start_index < min_length:
            break
        yield {
            "spec": spec[:, start_index:end_index],
            "pianoroll_actives": pianoroll["actives"][:, start_index:end_index],
            "pianoroll_onsets": pianoroll["onsets"][:, start_index:end_index],
            "pianoroll_offsets": pianoroll["offsets"][:, start_index:end_index],
            "pianoroll_velocities": pianoroll["velocities"][:, start_index:end_index],
        }


def prepare_dataset(input_path, output_path):
    splits_dirs = {
        "train": [
            "AkPnBcht/MUS",
            "AkPnBsdf/MUS",
            "AkPnCGdD/MUS",
            "AkPnStgb/MUS",
            "SptkBGAm/MUS",
            "SptkBGCl/MUS",
            "StbgTGd2/MUS",
        ],
        "validation": ["ENSTDkCl/MUS", "ENSTDkAm/MUS"],
    }

    split_wav_filenames = {}
    for split, dirs in splits_dirs.items():
        wav_globs = [os.path.join(input_path, d, "*.wav") for d in dirs]
        wav_filenames = [glob(g) for g in wav_globs]
        split_wav_filenames[split] = [f for d in wav_filenames for f in d]

    validation_ids = set(
        _id_from_filename(f) for f in split_wav_filenames["validation"]
    )
    split_wav_filenames["train"] = [
        f
        for f in split_wav_filenames["train"]
        if _id_from_filename(f) not in validation_ids
    ]

    for split, wav_filenames in split_wav_filenames.items():
        sample_dir = os.path.join(output_path, split)
        os.makedirs(sample_dir, exist_ok=True)
        for wav_filename in tqdm(wav_filenames, desc=f"preparing {split} set"):
            audio, _ = librosa.load(wav_filename, sr=constants.SAMPLE_RATE)
            spec = audio_to_spec(audio)

            midi_filenames = wav_filename[:-4] + ".mid"
            notesequence = load_notesequence(midi_filenames)
            pianoroll = notesequence_to_pianoroll(
                notesequence["pitches"],
                notesequence["intervals"],
                notesequence["velocities"],
            )
            pianoroll = {k: _set_length(v, spec.shape[1]) for k, v in pianoroll.items()}

            base_name = os.path.splitext(os.path.basename(wav_filename))[0]
            for part, sample in enumerate(
                _create_samples(
                    spec,
                    pianoroll,
                    constants.ACOUSTIC_TRAIN_MIN_LENGTH if split == "train" else None,
                    constants.ACOUSTIC_TRAIN_MAX_LENGTH if split == "train" else None,
                )
            ):
                np.savez(
                    os.path.join(sample_dir, f"{base_name}-part{part:03}.npz"), **sample
                )
