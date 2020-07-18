import numpy as np
import soundfile as sf
import librosa

from perfectpitch import constants


def load_audio(path):
    audio, sample_rate = sf.read(path, dtype=np.float32)
    audio = audio.mean(axis=1)
    audio = librosa.resample(
        audio, orig_sr=sample_rate, target_sr=constants.SAMPLE_RATE
    )
    return audio


def save_audio(path, audio):
    sf.write(path, audio, constants.SAMPLE_RATE)


def audio_to_spec(audio):
    return librosa.feature.melspectrogram(
        audio,
        sr=constants.SAMPLE_RATE,
        hop_length=constants.SPEC_HOP_LENGTH,
        fmin=30.0,
        n_mels=constants.SPEC_DIM,
        htk=True,
    ).T


def audio_to_posenc(audio, num_frames=None):
    frame_duration = constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE
    if num_frames is None:
        num_frames = int(len(audio) / frame_duration) + 1

    frames = np.expand_dims(np.arange(num_frames), axis=1)
    div_term = np.exp(
        np.arange(0, constants.POSENC_DIM, 2) * -np.log(10000) / constants.POSENC_DIM
    )
    points = frames * div_term
    sin = np.sin(points)
    cos = np.cos(points)
    return np.concatenate([sin, cos], axis=1).astype(np.float32)
