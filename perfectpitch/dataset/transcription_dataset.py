import os
from glob import glob

import torch


class TranscriptionDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        sample_filenames = sorted(glob(os.path.join(path, "*.pt")))
        if len(sample_filenames) == 0:
            raise ValueError(f"No sample found in {path}")

        self.__sample_filenames = sample_filenames

    def __len__(self):
        return len(self.__sample_filenames)

    def __getitem__(self, index):
        sample = torch.load(self.__sample_filenames[index])
        return {
            "audio": sample["audio"],
            "notesequence": {
                "pitches": sample["pitches"],
                "intervals": sample["intervals"],
                "velocities": sample["velocities"],
            },
        }
