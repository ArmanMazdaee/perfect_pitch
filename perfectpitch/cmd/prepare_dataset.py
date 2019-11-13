import sys
from glob import glob

import numpy as np
import h5py

try:
    import tensorflow as tf

    tf.compat.v1.enable_eager_execution()

    from magenta.music import sequences_lib
    from magenta.music import audio_io
    from magenta.protobuf import music_pb2
except ImportError:
    sys.exit("You need to install perfectpitch with [prepare_dataset]")

from perfectpitch import constants


def _parse_example(example_proto):
    features = {
        "id": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
        "audio": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
        "velocity_range": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
        "sequence": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
    }
    return tf.io.parse_single_example(example_proto, features)


def prepare_dataset(input_pattern, output_path):
    filenames = sorted(glob(input_pattern))
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(_parse_example)

    with h5py.File(output_path, "w") as f:
        f.attrs["sample_rate"] = constants.SAMPLE_RATE

        for index, example in enumerate(dataset.take(3)):
            example_id = example["id"].numpy().decode("utf-8")

            audio = audio_io.wav_data_to_samples(
                example["audio"].numpy(), constants.SAMPLE_RATE
            )

            velocity_range = music_pb2.VelocityRange.FromString(
                example["velocity_range"].numpy()
            )
            velocity_min = velocity_range.min
            velocity_max = velocity_range.max

            sequence = music_pb2.NoteSequence.FromString(example["sequence"].numpy())
            sequence = sequences_lib.apply_sustain_control_changes(sequence)
            pitches = np.array([note.pitch for note in sequence.notes], dtype=np.int8)
            intervals = np.array(
                [(note.start_time, note.end_time) for note in sequence.notes],
                dtype=np.float32,
            )
            velocities = np.array(
                [note.velocity for note in sequence.notes], dtype=np.int8
            )

            group = f.create_group("{:09}".format(index))
            group.attrs["id"] = example_id
            group.create_dataset("audio", data=audio, dtype="float32")
            group.create_dataset("velocity_min", data=velocity_min, dtype="int8")
            group.create_dataset("velocity_max", data=velocity_max, dtype="int8")
            group.create_dataset("pitches", data=pitches, dtype="int8")
            group.create_dataset("intervals", data=intervals, dtype="float32")
            group.create_dataset("velocities", data=velocities, dtype="int8")
