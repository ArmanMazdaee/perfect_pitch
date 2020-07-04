import random
from collections import deque

import torch

from perfectpitch.utils.data import transcription_to_pianoroll
from .transcription_dataset import TranscriptionDataset


class PianorollDataset(TranscriptionDataset):
    def __init__(self, path, shuffle, min_length=None, max_length=None, buffer_size=64):
        super().__init__(path, shuffle)
        self._min_length = min_length
        self._max_length = max_length
        self._buffer_size = buffer_size

    def _get_splits(self, length):
        splits = []
        min_length = 0 if self._min_length is None else self._min_length
        step = length if self._max_length is None else self._max_length
        for start in range(0, length, step):
            end = min(length, start + step)
            if end - start >= min_length:
                splits.append((start, end))
        return splits

    def __len__(self):
        total = 0
        for sample in super().__iter__():
            spec = sample["spec"]
            length = spec.shape[0]
            splits = self._get_splits(length)
            total += len(splits)

        return total

    def __iter__(self):
        buffer_size = self._buffer_size if self._shuffle else 1
        buffer = deque()
        for sample in super().__iter__():
            spec = sample["spec"]
            transcription = sample["transcription"]
            length = spec.shape[0]
            pianoroll = transcription_to_pianoroll(
                transcription["pitches"],
                transcription["intervals"],
                transcription["velocities"],
                length,
            )
            mask = torch.ones(length, dtype=torch.bool)

            for start, end in self._get_splits(length):
                buffer.append(
                    {
                        "spec": spec[start:end, :],
                        "pianoroll": {
                            key: value[start:end, :] for key, value in pianoroll.items()
                        },
                        "mask": mask[start:end],
                    }
                )

            if self._shuffle:
                random.shuffle(buffer)

            while len(buffer) > buffer_size:
                yield buffer.popleft()

        yield from buffer
