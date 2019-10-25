from glob import glob
import tensorflow as tf

from perfectpitch import constants
from perfectpitch.data import utils


def _parse_example(example_proto):
    features = {
        "id": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
        "sequence": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
        "audio": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
        "velocity_range": tf.io.FixedLenFeature(shape=(), dtype=tf.dtypes.string),
    }
    return tf.io.parse_single_example(example_proto, features)


def _process_example(example):
    audio = tf.numpy_function(
        utils.audio_from_string,
        [example["audio"]],
        tf.dtypes.float32,
        name="deserialize_audio",
    )
    audio.set_shape([None])

    velocity_min, velocity_max = tf.numpy_function(
        utils.velocity_range_from_string,
        [example["velocity_range"]],
        [tf.dtypes.int8, tf.dtypes.int8],
        name="deserialize_velocity_range",
    )
    velocity_min.set_shape([])
    velocity_max.set_shape([])

    pitches, intervals, velocities = tf.numpy_function(
        utils.notesequence_from_string,
        [example["sequence"]],
        [tf.dtypes.int8, tf.dtypes.float32, tf.dtypes.int8],
        name="deserialize_notesequence",
    )
    pitches.set_shape([None])
    intervals.set_shape([None, 2])
    velocities.set_shape([None])

    return {
        "id": example["id"],
        "audio": audio,
        "velocity_min": velocity_min,
        "velocity_max": velocity_max,
        "notesequence": {
            "pitches": pitches,
            "intervals": intervals,
            "velocities": velocities,
        },
    }


def _add_spec(data):
    spec = tf.numpy_function(
        utils.spec_from_audio, [data["audio"]], tf.dtypes.float32, name="audio_spec"
    )
    spec.set_shape([None, constants.SPEC_N_BINS])
    data["spec"] = spec
    return spec


def _element_spec_to_shape(element_spec):
    if isinstance(element_spec, dict):
        return {k: _element_spec_to_shape(v) for k, v in element_spec.items()}
    return element_spec.shape


def load_dataset(pattern, is_training, batch_size):
    filenames = sorted(glob(pattern))

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if is_training:
        dataset = dataset.shuffle(1000)

    dataset = dataset.interleave(
        map_func=tf.data.TFRecordDataset,
        cycle_length=tf.data.experimental.AUTOTUNE if is_training else 1,
    )

    if is_training:
        dataset = dataset.shuffle(10 * batch_size)

    dataset = dataset.map(_parse_example)
    dataset = dataset.map(_process_example)
    dataset = dataset.map(_add_spec)

    padded_shapes = _element_spec_to_shape(dataset.element_spec)
    return dataset.padded_batch(batch_size, padded_shapes, drop_remainder=is_training)
