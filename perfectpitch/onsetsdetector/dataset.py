import random

import torch

from perfectpitch.dataset.transcription_dataset import TranscriptionDataset
from perfectpitch.utils.data import transcription_to_pianoroll


class OnsetsDataset(TranscriptionDataset):
    def __init__(self, path, shuffle, min_length=None, max_length=None, buffer_size=64):
        super().__init__(path, shuffle)
        self._min_length = min_length
        self._max_length = max_length
        self._buffer_size = buffer_size

    def __iter__(self):
        min_length = 0 if self._min_length is None else self._min_length
        buffer_size = self._buffer_size if self._shuffle else 1
        buffer = []

        for spec, transcription in super().__iter__():
            length = spec.shape[1]
            pianoroll = transcription_to_pianoroll(
                transcription["pitches"],
                transcription["intervals"],
                transcription["velocities"],
                length,
            )
            onsets = pianoroll["onsets"]
            weights = torch.ones_like(onsets)
            weights[onsets == 1] = 2

            step = length if self._max_length is None else self._max_length
            for start in range(0, length, step):
                end = min(length, start + step)
                if end - start < min_length:
                    break

                buffer.append(
                    (spec[:, start:end], onsets[:, start:end], weights[:, start:end])
                )

            if self._shuffle:
                random.shuffle(buffer)

            while len(buffer) > buffer_size:
                yield buffer.pop()

        yield from buffer
