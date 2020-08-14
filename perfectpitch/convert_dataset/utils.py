import os

import tensorflow as tf
import note_seq
from tqdm import tqdm

from perfectpitch.utils import load_spec


def _convert_sample_py(name, wav_filename, midi_filename):
    with tf.io.gfile.GFile(wav_filename, "rb") as wav_file:
        spec = load_spec(wav_file)
    spec = tf.io.serialize_tensor(spec).numpy()

    with tf.io.gfile.GFile(midi_filename, "rb") as midi_file:
        midi = midi_file.read()
    notesequence = note_seq.midi_io.midi_to_note_sequence(midi)
    notesequence = note_seq.sequences_lib.apply_sustain_control_changes(notesequence)
    notesequence = notesequence.SerializeToString()

    feature = {
        "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
        "spec": tf.train.Feature(bytes_list=tf.train.BytesList(value=[spec])),
        "notesequence": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[notesequence])
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def _convert_sample(name, wav_filename, midi_filename):
    serialized_sample = tf.numpy_function(
        func=_convert_sample_py,
        inp=[name, wav_filename, midi_filename],
        Tout=tf.dtypes.string,
    )
    serialized_sample.set_shape([])
    return serialized_sample


def convert_dataset(output_path, split, names, wav_filenames, midi_filenames):
    dataset = tf.data.Dataset.from_tensor_slices((names, wav_filenames, midi_filenames))
    dataset = dataset.map(
        _convert_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    tf.io.gfile.makedirs(output_path)
    output_filename = os.path.join(output_path, f"{split}.tfrecord")
    with tf.io.TFRecordWriter(output_filename) as writer:
        for serialized_sample in tqdm(
            dataset, desc=f"converting {split} split", total=len(names)
        ):
            writer.write(serialized_sample.numpy())
