import torch

from perfectpitch.dataset.transcription_dataset import TranscriptionDataset
from perfectpitch.utils.data import notesequence_to_pianoroll


class AcousticDataset(TranscriptionDataset):
    def __init__(self, path, min_length=None, max_length=None, pad_sequences=False):
        super().__init__(path)

        if max_length is None and pad_sequences:
            raise ValueError("pad_sequences could not be true while max_length is None")
        self.__sequences_length = max_length if pad_sequences else None

        sample_splits = []
        for index in range(super().__len__()):
            sample = super().__getitem__(index)
            length = sample["spec"].shape[1]
            step = length if max_length is None else max_length
            for start in range(0, length, step):
                end = min(length, start + step)
                if min_length is None or end - start >= min_length:
                    sample_splits.append((index, start, end))
        self.__sample_splits = sample_splits

    def __len__(self):
        return len(self.__sample_splits)

    def __getitem__(self, index):
        sample_index, start, end = self.__sample_splits[index]
        sample = super().__getitem__(sample_index)
        spec = sample["spec"]
        length = end - start
        sample_length = spec.shape[1]

        notesequence = {
            key: value.numpy() for key, value in sample["notesequence"].items()
        }
        pianoroll = notesequence_to_pianoroll(
            notesequence["pitches"],
            notesequence["intervals"],
            notesequence["velocities"],
            sample_length,
        )
        pianoroll = {key: torch.from_numpy(value) for key, value in pianoroll.items()}

        if length < sample_length:
            spec = spec[:, start:end]
            pianoroll = {key: value[:, start:end] for key, value in pianoroll.items()}

        if self.__sequences_length is not None and self.__sequences_length > length:
            pad = (0, self.__sequences_length - length)
            spec = torch.nn.functional.pad(spec, pad)
            pianoroll = {
                key: torch.nn.functional.pad(value, pad)
                for key, value in pianoroll.items()
            }

        return {"spec": spec, "pianoroll": pianoroll}
