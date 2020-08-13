import os
import re

import tensorflow as tf

from .utils import convert_dataset


def _id_from_filename(filename):
    return re.match(r".*MAPS_MUS-(.*)_[^_]+\.[^.]+", filename).group(1)


def convert_maps(input_path, output_path):
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
        split_wav_filenames[split] = sorted(
            tf.io.gfile.glob([os.path.join(input_path, d, "*.wav") for d in dirs])
        )

    validation_ids = set(
        _id_from_filename(f) for f in split_wav_filenames["validation"]
    )
    split_wav_filenames["train"] = [
        f
        for f in split_wav_filenames["train"]
        if _id_from_filename(f) not in validation_ids
    ]

    split_names = {
        split: [os.path.basename(wav_filename[:-4]) for wav_filename in wav_filenames]
        for split, wav_filenames in split_wav_filenames.items()
    }

    split_midi_filenames = {
        split: [wav_filename[:-4] + ".mid" for wav_filename in wav_filenames]
        for split, wav_filenames in split_wav_filenames.items()
    }

    for split in ["train", "validation"]:
        convert_dataset(
            output_path,
            split,
            split_names[split],
            split_wav_filenames[split],
            split_midi_filenames[split],
        )
