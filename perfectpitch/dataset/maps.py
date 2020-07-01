import os
import re
from glob import glob

from .convert_dataset import convert_dataset


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
        wav_globs = [os.path.join(input_path, d, "*.wav") for d in dirs]
        split_wav_filenames[split] = [
            wav_filename for wav_glob in wav_globs for wav_filename in glob(wav_glob)
        ]

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

    for split in ["train", "validation"]:
        convert_dataset(
            os.path.join(output_path, split),
            split_names[split],
            split_wav_filenames[split],
            split_midi_filenames[split],
        )
