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


def _add_pianoroll(data):
    def func(pitches, intervals, velocities, max_velocity):
        pianoroll = utils.notesequence_to_pianoroll(
            pitches, intervals, velocities, max_velocity
        )
        return (
            pianoroll["actives"],
            pianoroll["onsets"],
            pianoroll["offsets"],
            pianoroll["velocities"],
        )

    notesequence = data["notesequence"]
    velocity_range = data["velocity_range"]
    actives, onsets, offsets, velocities = tf.numpy_function(
        func,
        [
            notesequence["pitches"],
            notesequence["intervals"],
            notesequence["velocities"],
            velocity_range["max"],
        ],
        [tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32],
        name="pianoroll",
    )
    num_piches = constants.MAX_PITCH - constants.MIN_PITCH + 1
    actives.set_shape([None, num_piches])
    onsets.set_shape([None, num_piches])
    offsets.set_shape([None, num_piches])
    velocities.set_shape([None, num_piches])
    data["pianoroll"] = {
        "actives": actives,
        "onsets": onsets,
        "offsets": offsets,
        "velocities": velocities,
    }
    return data


def _pad_or_truncate_pianoroll(data):
    def func(spec, actives, onsets, offsets, velocities):
        spec_length = spec.shape[0]
        pianoroll_length = actives.shape[0]
        if spec_length > pianoroll_length:
            paddings = [[0, spec_length - pianoroll_length], [0, 0]]
            return (
                tf.pad(actives, paddings),
                tf.pad(onsets, paddings),
                tf.pad(offsets, paddings),
                tf.pad(velocities, paddings),
            )
        elif spec_length < pianoroll_length:
            return (
                actives[:spec_length, :],
                onsets[:spec_length, :],
                offsets[:spec_length, :],
                velocities[:spec_length, :],
            )
        else:
            return actives, onsets, offsets, velocities

    pianoroll = data["pianoroll"]
    actives, onsets, offsets, velocities = tf.numpy_function(
        func,
        [
            data["spec"],
            pianoroll["actives"],
            pianoroll["onsets"],
            pianoroll["offsets"],
            pianoroll["velocities"],
        ],
        [tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32],
        name="pad_or_truncate_pianoroll",
    )
    num_piches = constants.MAX_PITCH - constants.MIN_PITCH + 1
    actives.set_shape([None, num_piches])
    onsets.set_shape([None, num_piches])
    offsets.set_shape([None, num_piches])
    velocities.set_shape([None, num_piches])
    data["pianoroll"] = {
        "actives": actives,
        "onsets": onsets,
        "offsets": offsets,
        "velocities": velocities,
    }
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
    pianoroll=False,
    pad_or_truncate_pianoroll=True,
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

    if pianoroll:
        dataset = dataset.map(_add_pianoroll)

    if spec and pianoroll and pad_or_truncate_pianoroll:
        dataset = dataset.map(_pad_or_truncate_pianoroll)

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
