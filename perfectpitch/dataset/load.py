import os

import tensorflow as tf

from perfectpitch import constants


def _parse_sample(serialized):
    features_description = {
        "name": tf.io.FixedLenFeature([], tf.dtypes.string),
        "spec": tf.io.FixedLenFeature([], tf.dtypes.string),
        "notesequence_pitches": tf.io.FixedLenFeature([], tf.dtypes.string),
        "notesequence_intervals": tf.io.FixedLenFeature([], tf.dtypes.string),
        "notesequence_velocities": tf.io.FixedLenFeature([], tf.dtypes.string),
    }
    features = tf.io.parse_single_example(serialized, features_description)

    name = features["name"]
    spec = tf.io.parse_tensor(features["spec"], tf.dtypes.float32)
    spec.set_shape([None, constants.SPEC_N_BINS])
    notesequence_pitches = tf.io.parse_tensor(
        features["notesequence_pitches"], tf.dtypes.int8
    )
    notesequence_pitches.set_shape([None])
    notesequence_intervals = tf.io.parse_tensor(
        features["notesequence_intervals"], tf.dtypes.float32
    )
    notesequence_intervals.set_shape([None, 2])
    notesequence_velocities = tf.io.parse_tensor(
        features["notesequence_velocities"], tf.dtypes.int8
    )
    notesequence_velocities.set_shape([None])

    return {
        "name": name,
        "spec": spec,
        "notesequence": {
            "pitches": notesequence_pitches,
            "intervals": notesequence_intervals,
            "velocities": notesequence_velocities,
        },
    }


def load_dataset(path, split):
    filename = os.path.join(path, split + ".tfrecord")
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_sample)
    return dataset
