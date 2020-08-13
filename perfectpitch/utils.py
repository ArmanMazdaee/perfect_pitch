from perfectpitch import constants

import librosa


def load_spec(path):
    audio, _ = librosa.load(path, sr=constants.SAMPLE_RATE)
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=constants.SAMPLE_RATE,
        hop_length=constants.SPEC_HOP_LENGTH,
        fmin=30.0,
        n_mels=constants.SPEC_DIM,
        htk=True,
    )
    return spec.T
