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
    win_length = 1024
    audio_length = len(audio)
    before_pad = (win_length - constants.SPEC_HOP_LENGTH) // 2
    after_pad = win_length - before_pad
    audio = np.pad(audio, (before_pad, after_pad), "reflect")

    pitches = np.arange(constants.MIN_PITCH, constants.MAX_PITCH + 1, dtype=np.float32)
    frequencies = 440 * (2 ** ((pitches - 69) / 12.0))
    points = np.linspace(
        start=0,
        stop=frequencies * 2 * np.pi * win_length / constants.SAMPLE_RATE,
        num=win_length,
        endpoint=False,
        axis=1,
    )
    sins = np.sin(points) + 1j * np.sin(points + 0.5 * np.pi)
    kernel = sins * np.hanning(win_length).astype(np.float32)

    spec = np.zeros(
        (audio_length // constants.SPEC_HOP_LENGTH, len(frequencies)), dtype=np.float32
    )
    for index in range(len(spec)):
        start = index * constants.SPEC_HOP_LENGTH
        end = start + win_length
        bins = (audio[start:end] * kernel).sum(axis=1)
        spec[index] = np.abs(bins)
    return spec


def spec_to_audio(spec):
    pitches = np.arange(constants.MIN_PITCH, constants.MAX_PITCH + 1, dtype=np.float32)
    frequencies = 440 * (2 ** ((pitches - 69) / 12.0))
    step = frequencies * 2 * np.pi * constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE

    audio = np.zeros((len(spec) - 1) * constants.SPEC_HOP_LENGTH, dtype=np.float32)
    for index in range(len(spec) - 1):
        points = np.linspace(
            start=index * step,
            stop=(index + 1) * step,
            num=constants.SPEC_HOP_LENGTH,
            endpoint=False,
        )
        sins = np.sin(points)
        weights = np.linspace(
            start=spec[index],
            stop=spec[index + 1],
            num=constants.SPEC_HOP_LENGTH,
            endpoint=False,
        )
        start = index * constants.SPEC_HOP_LENGTH
        end = (index + 1) * constants.SPEC_HOP_LENGTH
        audio[start:end] = (sins * weights).mean(axis=1)
    return audio


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
