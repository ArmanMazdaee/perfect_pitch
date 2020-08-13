import os
import json

import tensorflow as tf

from .utils import convert_dataset


def convert_maestro(input_path, output_path):
    info_filename = os.path.join(input_path, "maestro-v1.0.0.json")
    with tf.io.gfile.GFile(info_filename) as info_file:
        info = json.load(info_file)

    for split in ["train", "validation", "test"]:
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
            output_path, split, names, wav_filenames, midi_filenames,
        )
