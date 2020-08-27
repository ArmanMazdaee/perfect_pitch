import functools

import tensorflow as tf

from perfectpitch import constants


@tf.function
def _parse_example(example):
    features_description = {
        "spec": tf.io.FixedLenFeature([], tf.dtypes.string),
        "pitches": tf.io.FixedLenFeature([], tf.dtypes.string),
        "start_times": tf.io.FixedLenFeature([], tf.dtypes.string),
    }
    features = tf.io.parse_single_example(example, features_description)
    spec = tf.io.parse_tensor(features["spec"], tf.dtypes.float32)
    spec.set_shape([None, constants.SPEC_DIM])
    pitches = tf.io.parse_tensor(features["pitches"], tf.dtypes.int8)
    pitches.set_shape([None])
    start_times = tf.io.parse_tensor(features["start_times"], tf.dtypes.float32)
    start_times.set_shape([None])

    frame_duration = constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE
    num_frames = len(spec)
    pitches = tf.cast(pitches - constants.MIN_PITCH, tf.dtypes.int32)
    pitches = tf.concat([pitches, pitches], axis=0)
    onset_frames = tf.cast(start_times // frame_duration, tf.dtypes.int32)
    onset_frames = tf.concat([onset_frames, onset_frames + 1], axis=0)
    valid_indices = tf.math.logical_and(pitches >= 0, pitches < constants.NUM_PITCHES)
    valid_indices = tf.math.logical_and(valid_indices, onset_frames <= num_frames - 1)
    pitches = pitches[valid_indices]
    onset_frames = onset_frames[valid_indices]
    onsets = tf.scatter_nd(
        indices=tf.stack([onset_frames, pitches], axis=1),
        updates=tf.ones(tf.shape(pitches)),
        shape=(num_frames, constants.NUM_PITCHES),
    )
    onsets = tf.math.minimum(onsets, 1.0)

    return {
        "spec": spec,
        "onsets": onsets,
        "length": num_frames,
    }


@tf.function
def _split_example(example, min_length, max_length):
    spec = example["spec"]
    onsets = example["onsets"]
    length = example["length"]

    last_length = length % max_length
    length = tf.fill((length // max_length,), max_length)
    if 0 < last_length and last_length < min_length:
        spec = spec[:-last_length]
        onsets = onsets[:-last_length]
    elif last_length >= min_length:
        length = tf.concat((length, [last_length]), axis=0)
        padding = max_length - last_length
        spec = tf.pad(spec, [[0, padding], [0, 0]])
        onsets = tf.pad(onsets, [[0, padding], [0, 0]])

    return tf.data.Dataset.from_tensor_slices(
        {
            "spec": tf.reshape(spec, (-1, max_length, constants.SPEC_DIM)),
            "onsets": tf.reshape(onsets, (-1, max_length, constants.NUM_PITCHES)),
            "length": length,
        },
    )


def get_dataset(path, batch_size, shuffle, min_length, max_length):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(
        _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.flat_map(
        functools.partial(_split_example, min_length=min_length, max_length=max_length)
    )

    if shuffle:
        dataset = dataset.shuffle(batch_size * 10)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
