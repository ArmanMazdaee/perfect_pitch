import os

import tensorflow as tf
from tqdm import tqdm

from perfectpitch.utils.data import load_spec, load_transcription


def _create_example_py(name, wav_filename, midi_filename):
    spec = load_spec(wav_filename)
    spec = tf.io.serialize_tensor(spec).numpy()

    transcription = load_transcription(midi_filename)
    pitches = tf.io.serialize_tensor(transcription["pitches"]).numpy()
    start_times = tf.io.serialize_tensor(transcription["start_times"]).numpy()
    end_times = tf.io.serialize_tensor(transcription["end_times"]).numpy()
    velocities = tf.io.serialize_tensor(transcription["velocities"]).numpy()

    feature = {
        "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
        "spec": tf.train.Feature(bytes_list=tf.train.BytesList(value=[spec])),
        "pitches": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pitches])),
        "start_times": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[start_times])
        ),
        "end_times": tf.train.Feature(bytes_list=tf.train.BytesList(value=[end_times])),
        "velocities": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[velocities])
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def _create_example(name, wav_filename, midi_filename):
    example = tf.numpy_function(
        func=_create_example_py,
        inp=[name, wav_filename, midi_filename],
        Tout=tf.dtypes.string,
    )
    example.set_shape([])
    return example


def convert_dataset(output_path, split, names, wav_filenames, midi_filenames):
    dataset = tf.data.Dataset.from_tensor_slices((names, wav_filenames, midi_filenames))
    dataset = dataset.map(
        _create_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    tf.io.gfile.makedirs(output_path)
    output_filename = os.path.join(output_path, f"{split}.tfrecord")
    with tf.io.TFRecordWriter(output_filename) as writer:
        for example in tqdm(
            dataset, desc=f"converting {split} split", total=len(names)
        ):
            writer.write(example.numpy())
