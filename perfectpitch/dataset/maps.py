import os
import re

import tensorflow as tf

from .convert import convert_dataset


TRAIN_DIRS = [
    "AkPnBcht/MUS",
    "AkPnBsdf/MUS",
    "AkPnCGdD/MUS",
    "AkPnStgb/MUS",
    "SptkBGAm/MUS",
    "SptkBGCl/MUS",
    "StbgTGd2/MUS",
]
VALIDATION_DIRS = ["ENSTDkCl/MUS", "ENSTDkAm/MUS"]
TRAIN_NUM_SHARDS = 5
VALIDATION_NUM_SHARDS = 2


def _id_from_filename(filename):
    return re.match(r".*MAPS_MUS-(.*)_[^_]+\.[^.]+", filename).group(1)


def convert_maps(input_path, output_path):
    splits_dirs = {
        "train": TRAIN_DIRS,
        "validation": VALIDATION_DIRS,
    }

    split_wav_filenames = {}
    for split, dirs in splits_dirs.items():
        wav_globs = [os.path.join(input_path, d, "*.wav") for d in dirs]
        split_wav_filenames[split] = sorted(tf.io.gfile.glob(wav_globs))

    validation_ids = set(
        _id_from_filename(f) for f in split_wav_filenames["validation"]
    )
    split_wav_filenames["train"] = [
        f
        for f in split_wav_filenames["train"]
        if _id_from_filename(f) not in validation_ids
    ]

    split_midi_filenames = {
        split: [wav_filename[:-4] + ".mid" for wav_filename in wav_filenames]
        for split, wav_filenames in split_wav_filenames.items()
    }

    split_names = {
        split: [os.path.basename(wav_filename[:-4]) for wav_filename in wav_filenames]
        for split, wav_filenames in split_wav_filenames.items()
    }

    split_num_shards = {"train": TRAIN_NUM_SHARDS, "validation": VALIDATION_NUM_SHARDS}

    for split, num_shards in split_num_shards.items():
        convert_dataset(
            split_names[split],
            split_wav_filenames[split],
            split_midi_filenames[split],
            output_path,
            split,
            num_shards,
        )
