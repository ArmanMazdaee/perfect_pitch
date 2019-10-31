import operator
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

        SUSTAIN_ON = 0
        SUSTAIN_OFF = 1
        NOTE_ON = 2
        NOTE_OFF = 3

        events = []
        events.extend(
            [(NOTE_ON, note.start_time, note) for note in transcription.notes]
        )
        events.extend([(NOTE_OFF, note.end_time, note) for note in transcription.notes])
        events.extend(
            [
                (
                    SUSTAIN_ON if control_change.control_value >= 64 else SUSTAIN_OFF,
                    control_change.time,
                    control_change,
                )
                for control_change in transcription.control_changes
                if control_change.control_number == 64
            ]
        )
        events.sort(key=operator.itemgetter(1, 0))

        active_notes = []
        sustain = False
        for kind, time, meta in events:
            if kind == SUSTAIN_ON and not sustain:
                sustain = True
            elif kind == SUSTAIN_OFF and sustain:
                sustain = False
                for note in list(active_notes):
                    if note.end_time < time:
                        note.end_time = time
                        active_notes.remove(note)
            elif kind == NOTE_ON:
                if sustain:
                    for note in list(active_notes):
                        if note.pitch == meta.pitch:
                            note.end_time = time
                            active_notes.remove(note)
                            if note.start_time == note.end_time:
                                transcription.notes.remove(note)
                active_notes.append(meta)
            elif kind == NOTE_OFF and not sustain:
                if meta in active_notes:
                    active_notes.remove(meta)

        for note in active_notes:
            note.end_time = time

        pitches = np.array([note.pitch for note in transcription.notes], dtype=np.int8)
        intervals = np.array(
            [(note.start_time, note.end_time) for note in transcription.notes],
            dtype=np.float32,
        )
        velocities = np.array(
            [note.velocity for note in transcription.notes], dtype=np.int8
        )
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
        utils.audio_to_spec, [data["audio"]], tf.dtypes.float32, name="audio_spec"
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
