import os
from glob import glob

import numpy as np
import torch


class TranscriptionDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        sample_filenames = sorted(glob(os.path.join(path, "*.npz")))
        if len(sample_filenames) == 0:
            raise ValueError(f"No sample found in {path}")

        self.__sample_filenames = sample_filenames

    def __len__(self):
        return len(self.__sample_filenames)

    def __getitem__(self, index):
        sample = np.load(self.__sample_filenames[index])
        return {
            "spec": torch.from_numpy(sample["spec"]),
            "notesequence": {
                "pitches": torch.from_numpy(sample["notesequence_pitches"]),
                "intervals": torch.from_numpy(sample["notesequence_intervals"]),
                "velocities": torch.from_numpy(sample["notesequence_velocities"]),
            },
        }
