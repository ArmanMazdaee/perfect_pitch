import numpy as np
import torch
import h5py

from perfectpitch import constants
from perfectpitch.data import utils


def _set_length(tensor, length):
    if tensor.shape[1] > length:
        return tensor[:, :length]
    elif tensor.shape[1] < length:
        return np.pad(tensor, [(0, 0), (0, length - tensor.shape[1])])
    return tensor


def _get_pianoroll_weight(pianoroll):
    onsets = np.zeros_like(pianoroll["onsets"])
    onsets.fill(0.01)
    onsets[pianoroll["onsets"] > 0.5] = 0.99

    offsets = np.zeros_like(pianoroll["offsets"])
    offsets.fill(0.01)
    offsets[pianoroll["offsets"] > 0.5] = 0.99

    actives = np.zeros_like(pianoroll["actives"])
    actives.fill(0.05)
    actives[pianoroll["actives"] > 0.5] = 0.95

    velocities = np.zeros_like(pianoroll["velocities"])
    velocities.fill(0)
    velocities[pianoroll["onsets"] > 0.5] = 1

    return {
        "onsets": onsets,
        "offsets": offsets,
        "actives": actives,
        "velocities": velocities,
    }


def _numpy_to_pytorch(data):
    if isinstance(data, dict):
        return {key: _numpy_to_pytorch(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    raise TypeError(f"{type(data)} is not supported")


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        audio=False,
        spec=False,
        spec_length=False,
        velocity_min=False,
        velocity_max=False,
        notesequence=False,
        pianoroll=False,
        pianoroll_weight=False,
        pad_or_truncate_pianoroll=True,
    ):
        self.__path = path
        self.__audio = audio
        self.__spec = spec
        self.__spec_length = spec_length
        self.__velocity_min = velocity_min
        self.__velocity_max = velocity_max
        self.__notesequence = notesequence
        self.__pianoroll = pianoroll
        self.__pianoroll_weight = pianoroll_weight
        self.__pad_or_truncate_pianoroll = pad_or_truncate_pianoroll

        with h5py.File(path, "r") as f:
            if f.attrs["sample_rate"] != constants.SAMPLE_RATE:
                raise ValueError(
                    "dataset sample rate does not match with the model sample rate"
                )

            self.__examples = sorted(f.keys())

    def __len__(self):
        return len(self.__examples)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        with h5py.File(self.__path, "r") as f:
            example = f[self.__examples[index]]
            audio = example["audio"][...]
            velocity_min = example["velocity_min"][...]
            velocity_max = example["velocity_max"][...]
            pitches = example["pitches"][...]
            intervals = example["intervals"][...]
            velocities = example["velocities"][...]

        data = {}
        if self.__audio:
            data["audio"] = audio

        if self.__velocity_min:
            data["velocity_min"] = velocity_min

        if self.__velocity_max:
            data["velocity_max"] = velocity_max

        if self.__notesequence:
            notesequence = {}
            notesequence["pitches"] = pitches
            notesequence["intervals"] = intervals
            notesequence["velocities"] = velocities
            data["notesequence"] = notesequence

        if self.__spec:
            data["spec"] = utils.audio_to_spec(audio)

        spec_length = int(len(audio) / constants.SPEC_HOP_LENGTH) + 1
        if self.__spec_length:
            data["spec_length"] = np.array(spec_length)

        if self.__pianoroll or self.__pianoroll_weight:
            pianoroll = utils.notesequence_to_pianoroll(
                pitches, intervals, velocities, velocity_max
            )
            if self.__pad_or_truncate_pianoroll:
                pianoroll = {
                    key: _set_length(value, spec_length)
                    for key, value in pianoroll.items()
                }
            if self.__pianoroll:
                data["pianoroll"] = pianoroll
            if self.__pianoroll_weight:
                data["pianoroll_weight"] = _get_pianoroll_weight(pianoroll)

        return _numpy_to_pytorch(data)
