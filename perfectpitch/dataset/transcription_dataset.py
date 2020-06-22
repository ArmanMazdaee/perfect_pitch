import os
import random
from glob import glob

import torch


class TranscriptionDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, shuffle):
        self._path = path
        self._shuffle = shuffle

    def __iter__(self):
        sample_filenames = sorted(glob(os.path.join(self._path, "*.pt")))
        if len(sample_filenames) == 0:
            raise RuntimeError(f"No sample found in {self._path}")

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            sample_filenames = [
                f
                for i, f in enumerate(sample_filenames)
                if i % worker_info.num_workers == worker_info.id
            ]

        if self._shuffle:
            random.shuffle(sample_filenames)

        for sample_filename in sample_filenames:
            sample = torch.load(sample_filename)
            yield (
                sample["spec"],
                {
                    "pitches": sample["pitches"],
                    "intervals": sample["intervals"],
                    "velocities": sample["velocities"],
                },
            )
