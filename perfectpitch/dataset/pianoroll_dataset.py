import numpy as np

from perfectpitch.utils.transcription import transcription_to_pianoroll
from .transcription_dataset import TranscriptionDataset


class PianorollDataset(TranscriptionDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        spec = sample["spec"]
        length = spec.shape[0]
        transcription = sample["transcription"]
        pianoroll = transcription_to_pianoroll(
            transcription["pitches"],
            transcription["intervals"],
            transcription["velocities"],
            length,
        )
        mask = np.ones(length, dtype=np.bool)

        return {
            "spec": spec,
            "pianoroll": pianoroll,
            "mask": mask,
        }
