import os
from glob import glob

import numpy as np
import torch


class TranscriptionDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        sample_filenames = sorted(glob(os.path.join(path, "*.npz")))
        if len(sample_filenames) == 0:
            raise ValueError(f"No sample found in {path}")

        self._sample_filenames = sample_filenames

    def __len__(self):
        return len(self._sample_filenames)

    def __getitem__(self, index):
        sample_filename = self._sample_filenames[index]
        sample = np.load(sample_filename)
        return {
            "spec": sample["spec"],
            "transcription": {
                "pitches": sample["pitches"],
                "intervals": sample["intervals"],
                "velocities": sample["velocities"],
            },
        }
