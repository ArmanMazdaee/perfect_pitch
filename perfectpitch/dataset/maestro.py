import os
import json

import tensorflow as tf

from .convert import convert_dataset


TRAIN_NUM_SHARDS = 30
VALIDATION_NUM_SHARDS = 4
TEST_NUM_SHARDS = 4


def convert_maestro(input_path, output_path):
    info_filename = os.path.join(input_path, "maestro-v1.0.0.json")
    with tf.io.gfile.GFile(info_filename) as file:
        info = json.load(file)

    split_num_shards = {
        "train": TRAIN_NUM_SHARDS,
        "validation": VALIDATION_NUM_SHARDS,
        "test": TEST_NUM_SHARDS,
    }

    for split, num_shards in split_num_shards.items():
        names = [
            sample["audio_filename"][:-4] for sample in info if sample["split"] == split
        ]
        wav_filenames = [
            os.path.join(input_path, sample["audio_filename"])
            for sample in info
            if sample["split"] == split
        ]
        midi_filenames = [
            os.path.join(input_path, sample["midi_filename"])
            for sample in info
            if sample["split"] == split
        ]
        convert_dataset(
            names, wav_filenames, midi_filenames, output_path, split, num_shards
        )
