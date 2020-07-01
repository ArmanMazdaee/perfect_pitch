import os
import json

from .convert_dataset import convert_dataset


def convert_maestro(input_path, output_path):
    info_filename = os.path.join(input_path, "maestro-v1.0.0.json")
    with open(info_filename) as file:
        info = json.load(file)

    for split in ["train", "validation", "test"]:
        names = [
            sample["audio_filename"][5:-4]
            for sample in info
            if sample["split"] == split
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
            os.path.join(output_path, split), names, wav_filenames, midi_filenames,
        )
