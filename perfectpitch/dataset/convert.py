import os
import contextlib

import tensorflow as tf
from tqdm import tqdm

from perfectpitch import constants
from perfectpitch.utils.data import load_spec, load_notesequence


def _load_notesequence(path):
    notesequence = load_notesequence(path)
    return (
        notesequence["pitches"],
        notesequence["intervals"],
        notesequence["velocities"],
    )


def _load_sample(sample):
    spec = tf.numpy_function(load_spec, [sample["wav_filename"]], tf.dtypes.float32)
    spec.set_shape([None, constants.SPEC_N_BINS])

    (
        notesequence_pitches,
        notesequence_intervals,
        notesequence_velocities,
    ) = tf.numpy_function(
        _load_notesequence,
        [sample["midi_filename"]],
        [tf.dtypes.int8, tf.dtypes.float32, tf.dtypes.int8],
    )
    notesequence_pitches.set_shape([None])
    notesequence_intervals.set_shape([None, 2])
    notesequence_velocities.set_shape([None])
    return {
        "name": sample["name"],
        "spec": spec,
        "notesequence_pitches": notesequence_pitches,
        "notesequence_intervals": notesequence_intervals,
        "notesequence_velocities": notesequence_velocities,
    }


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _serialize_sample(sample):
    features = tf.train.Features(
        feature={
            "name": _bytes_feature(sample["name"]),
            "spec": _bytes_feature(tf.io.serialize_tensor(sample["spec"])),
            "notesequence_pitches": _bytes_feature(
                tf.io.serialize_tensor(sample["notesequence_pitches"])
            ),
            "notesequence_intervals": _bytes_feature(
                tf.io.serialize_tensor(sample["notesequence_intervals"])
            ),
            "notesequence_velocities": _bytes_feature(
                tf.io.serialize_tensor(sample["notesequence_velocities"])
            ),
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def convert_dataset(
    names, wav_filenames, midi_filenames, output_path, split, num_shards
):
    dataset = tf.data.Dataset.from_tensor_slices(
        {"name": names, "wav_filename": wav_filenames, "midi_filename": midi_filenames}
    )
    dataset = dataset.map(_load_sample, tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    os.makedirs(output_path, exist_ok=True)
    with contextlib.ExitStack() as stack:
        filenames = [
            os.path.join(output_path, f"{split}-{index:02}.tfrecord")
            for index in range(1, num_shards + 1)
        ]
        writers = [
            stack.enter_context(tf.io.TFRecordWriter(filename))
            for filename in filenames
        ]

        for index, sample in enumerate(
            tqdm(dataset, desc=f"converting {split} set", total=len(names))
        ):
            writers[index % num_shards].write(_serialize_sample(sample))
