import functools

import tensorflow as tf

from perfectpitch import constants
from perfectpitch.utils.data import transcription_to_pianoroll


def _parse_example(example, augment):
    features_description = {
        "pitches": tf.io.FixedLenFeature([], tf.dtypes.string),
        "start_times": tf.io.FixedLenFeature([], tf.dtypes.string),
        "end_times": tf.io.FixedLenFeature([], tf.dtypes.string),
    }
    features = tf.io.parse_single_example(example, features_description)
    pitches = tf.io.parse_tensor(features["pitches"], tf.dtypes.int8)
    pitches.set_shape([None])
    start_times = tf.io.parse_tensor(features["start_times"], tf.dtypes.float32)
    start_times.set_shape([None])
    end_times = tf.io.parse_tensor(features["end_times"], tf.dtypes.float32)
    end_times.set_shape([None])

    onsets, offsets = tf.numpy_function(
        transcription_to_pianoroll,
        inp=[pitches, start_times, end_times, augment],
        Tout=[tf.dtypes.float32, tf.dtypes.float32],
    )
    onsets.set_shape([None, constants.NUM_PITCHES])
    offsets.set_shape([None, constants.NUM_PITCHES])

    length = len(onsets)
    mask = tf.random.uniform((length,)) <= 0.3

    return {
        "onsets": onsets,
        "offsets": offsets,
        "length": length,
        "mask": mask,
    }


def _split_example(example, max_length):
    row_starts = tf.range(example["length"], delta=max_length)
    onsets = tf.RaggedTensor.from_row_starts(example["onsets"], row_starts)
    offsets = tf.RaggedTensor.from_row_starts(example["offsets"], row_starts)
    length = onsets.row_lengths()
    mask = tf.RaggedTensor.from_row_starts(example["mask"], row_starts)
    return tf.data.Dataset.from_tensor_slices(
        {
            "onsets": onsets.to_tensor(),
            "offsets": offsets.to_tensor(),
            "length": length,
            "mask": mask.to_tensor(),
        }
    )


def _is_long_example(example, min_length):
    return example["length"] >= min_length


def get_dataset(
    path, batch_size, shuffle=False, augment=False, min_length=None, max_length=None
):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(
        functools.partial(_parse_example, augment=augment),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if max_length is not None:
        dataset = dataset.flat_map(
            functools.partial(_split_example, max_length=max_length)
        )

    if min_length is not None:
        dataset = dataset.filter(
            functools.partial(_is_long_example, min_length=min_length)
        )

    if shuffle:
        dataset = dataset.shuffle(batch_size * 10)

    dataset = dataset.padded_batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
