import random
from collections import deque

from perfectpitch.utils.data import transcription_to_pianoroll
from .transcription_dataset import TranscriptionDataset


class PianorollDataset(TranscriptionDataset):
    def __init__(self, path, shuffle, min_length=None, max_length=None, buffer_size=64):
        super().__init__(path, shuffle)
        self._min_length = min_length
        self._max_length = max_length
        self._buffer_size = buffer_size

    def __iter__(self):
        min_length = 0 if self._min_length is None else self._min_length
        buffer_size = self._buffer_size if self._shuffle else 1
        buffer = deque()

        for sample in super().__iter__():
            spec = sample["spec"]
            transcription = sample["transcription"]
            length = spec.shape[1]
            pianoroll = transcription_to_pianoroll(
                transcription["pitches"],
                transcription["intervals"],
                transcription["velocities"],
                length,
            )

            step = length if self._max_length is None else self._max_length
            for start in range(0, length, step):
                end = min(length, start + step)
                if end - start < min_length:
                    break

                buffer.append(
                    {
                        "spec": spec[:, start:end],
                        "pianoroll": {
                            key: value[:, start:end] for key, value in pianoroll.items()
                        },
                    }
                )

            if self._shuffle:
                random.shuffle(buffer)

            while len(buffer) > buffer_size:
                yield buffer.popleft()

        yield from buffer
