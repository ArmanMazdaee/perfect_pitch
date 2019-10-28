import operator
import math
import io
from glob import glob

import numpy as np
import librosa
import tensorflow as tf

from perfectpitch import constants
from perfectpitch.protobuf import music_pb2
from perfectpitch import utils


def _parse_audio(serialized):
    def func(serialized):
        audio_file = io.BytesIO(serialized)
        wav, _ = librosa.load(audio_file, constants.SAMPLE_RATE)
        return librosa.to_mono(wav)

    audio = tf.numpy_function(func, [serialized], tf.dtypes.float32, name="parse_audio")
    audio.set_shape([None])
    return audio


def _parse_velocity_range(serialized):
    def func(serialized):
        velocity_range = music_pb2.VelocityRange.FromString(serialized)
        return (
            np.array(velocity_range.min, dtype=np.int8),
            np.array(velocity_range.max, dtype=np.int8),
        )

    velocity_min, velocity_max = tf.numpy_function(
        func,
        [serialized],
        [tf.dtypes.int8, tf.dtypes.int8],
        name="parse_velocity_range",
    )
    velocity_min.set_shape([])
    velocity_max.set_shape([])
    return {"min": velocity_min, "max": velocity_max}


def _parse_notesequence(serialized):
    def func(serialized):
        transcription = music_pb2.NoteSequence.FromString(serialized)

        sustains_ranges = []
        sustain_start = None
        control_changes = sorted(
            transcription.control_changes, key=operator.attrgetter("time")
        )
        for control_change in control_changes:
            if control_change.control_number != 64:
                continue
            if sustain_start is None and control_change.control_value >= 64:
                sustain_start = control_change.time
            elif sustain_start is not None and control_change.control_value < 64:
                sustains_ranges.append((sustain_start, control_change.time))
                sustain_start = None
        if sustain_start is not None:
            sustain_end = max(sustain_start, transcription.total_time)
            sustains_ranges.append((sustain_start, sustain_end))
        sustains_ranges.append((math.inf, math.inf))

        active_sustain = 0
        sustain_start, sustain_end = sustains_ranges[active_sustain]
        notes = sorted(transcription.notes, key=operator.attrgetter("end_time"))
        pitches = np.zeros([len(notes)], dtype=np.int8)
        intervals = np.zeros([len(notes), 2], dtype=np.float32)
        velocities = np.zeros([len(notes)], dtype=np.int8)

        for i, note in enumerate(notes):
            while note.end_time >= sustain_end:
                active_sustain += 1
                sustain_start, sustain_end = sustains_ranges[active_sustain]

            pitches[i] = note.pitch
            intervals[i, 0] = note.start_time
            if sustain_start < note.end_time < sustain_end:
                intervals[i, 1] = sustain_end
            else:
                intervals[i, 1] = note.end_time
            velocities[i] = note.velocity

        return pitches, intervals, velocities

    pitches, intervals, velocities = tf.numpy_function(
        func,
        [serialized],
        [tf.dtypes.int8, tf.dtypes.float32, tf.dtypes.int8],
        name="parse_notesequence",
    )
    pitches.set_shape([None])
    intervals.set_shape([None, 2])
    velocities.set_shape([None])
    return {"pitches": pitches, "intervals": intervals, "velocities": velocities}


def _parse_example(example_proto):
    features = {
        "id": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
        "audio": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
        "velocity_range": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
        "sequence": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
    }
    example = tf.io.parse_single_example(example_proto, features)

    return {
        "id": example["id"],
        "audio": _parse_audio(example["audio"]),
        "velocity_range": _parse_velocity_range(example["velocity_range"]),
        "notesequence": _parse_notesequence(example["sequence"]),
    }


def _add_spec(data):
    spec = tf.numpy_function(
        utils.spec_from_audio, [data["audio"]], tf.dtypes.float32, name="audio_spec"
    )
    spec.set_shape([None, constants.SPEC_N_BINS])
    data["spec"] = spec
    return data


def _remove_data(key):
    def func(data):
        del data[key]
        return data

    return func


def _element_spec_to_shape(element_spec):
    if isinstance(element_spec, dict):
        return {k: _element_spec_to_shape(v) for k, v in element_spec.items()}
    return element_spec.shape


def load_dataset(
    pattern,
    batch_size,
    shuffle,
    drop_remainder,
    id=False,
    audio=False,
    velocity_range=False,
    notesequence=False,
    spec=False,
):
    filenames = sorted(glob(pattern))

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.interleave(
        map_func=tf.data.TFRecordDataset,
        cycle_length=tf.data.experimental.AUTOTUNE if shuffle else 1,
    )

    if shuffle:
        dataset = dataset.shuffle(10 * batch_size)

    dataset = dataset.map(_parse_example)

    if spec:
        dataset = dataset.map(_add_spec)

    if not id:
        dataset = dataset.map(_remove_data("id"))

    if not audio:
        dataset = dataset.map(_remove_data("audio"))

    if not velocity_range:
        dataset = dataset.map(_remove_data("velocity_range"))

    if not notesequence:
        dataset = dataset.map(_remove_data("notesequence"))

    padded_shapes = _element_spec_to_shape(dataset.element_spec)
    return dataset.padded_batch(batch_size, padded_shapes, drop_remainder=shuffle)
