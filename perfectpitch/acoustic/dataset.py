import os
from glob import glob

import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        sample_glob = os.path.join(path, split, "*.npz")
        self.__sample_filenames = sorted(glob(sample_glob))
        if len(self.__sample_filenames) == 0:
            raise ValueError("Wrong dataset path or split")

    def __len__(self):
        return len(self.__sample_filenames)

    def __getitem__(self, index):
        sample = np.load(self.__sample_filenames[index])
        spec = torch.from_numpy(sample["spec"])
        pianoroll = {
            "actives": torch.from_numpy(sample["pianoroll_actives"]),
            "onsets": torch.from_numpy(sample["pianoroll_onsets"]),
            "offsets": torch.from_numpy(sample["pianoroll_offsets"]),
            "velocities": torch.from_numpy(sample["pianoroll_velocities"]),
        }
        return spec, pianoroll
