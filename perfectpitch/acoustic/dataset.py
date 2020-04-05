import tensorflow as tf

from perfectpitch import constants
from perfectpitch.dataset.load import load_dataset as load_transcription_dataset


MIN_LENGTH = 100
MAX_LENGTH = 2500
BATCH_SIZE_PER_REPLICA = 8
SHUFFLE_BUFFER_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA * 16


def _create_pianoroll__create_actives_indicies(pitches, onsets, offsets):
    length = tf.shape(pitches)[0]
    _, frame_indicies, pitch_indicies = tf.while_loop(
        cond=lambda i, frame_indicies, pitch_indicies: i < length,
        body=lambda i, frame_indicies, pitch_indicies: (
            i + 1,
            tf.concat([frame_indicies, tf.range(onsets[i], offsets[i])], axis=0),
            tf.concat(
                [pitch_indicies, tf.fill([offsets[i] - onsets[i]], pitches[i])], axis=0
            ),
        ),
        loop_vars=(
            tf.constant(0, tf.dtypes.int32),
            tf.zeros([0], tf.dtypes.int32),
            tf.zeros([0], tf.dtypes.int32),
        ),
        shape_invariants=(
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
        ),
        parallel_iterations=1,
    )
    return tf.stack([frame_indicies, pitch_indicies], axis=1)


def _create_pianoroll__create_frame(shape, indices):
    num_values = tf.shape(indices)[0]
    values = tf.fill([num_values], tf.constant(1, tf.dtypes.float32))
    return tf.scatter_nd(indices, values, shape)


def _create_pianoroll(notesequence, num_frames):
    frame_duration = constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE
    num_pitches = constants.MAX_PITCH - constants.MIN_PITCH + 1
    frame_shape = [num_frames, num_pitches]

    pitches = tf.cast(notesequence["pitches"] - constants.MIN_PITCH, tf.dtypes.int32)
    onsets = tf.cast(notesequence["intervals"][:, 0] / frame_duration, tf.dtypes.int32)
    onsets = tf.minimum(onsets, num_frames - 1)
    offsets = tf.cast(notesequence["intervals"][:, 1] / frame_duration, tf.dtypes.int32)
    offsets = tf.minimum(offsets, num_frames - 1)
    velocities = tf.cast(
        notesequence["velocities"] / tf.reduce_max(notesequence["velocities"]),
        tf.dtypes.float32,
    )

    valid_indices = pitches >= 0
    valid_indices = tf.logical_and(valid_indices, pitches < num_pitches)
    valid_indices = tf.logical_and(valid_indices, tf.not_equal(onsets, offsets))
    pitches = pitches[valid_indices]
    onsets = onsets[valid_indices]
    offsets = offsets[valid_indices]
    velocities = velocities[valid_indices]

    actives_indicies = _create_pianoroll__create_actives_indicies(
        pitches, onsets, offsets
    )
    actives_frame = _create_pianoroll__create_frame(frame_shape, actives_indicies)

    onsets_indices = tf.stack([onsets, pitches], axis=1)
    onsets_frame = _create_pianoroll__create_frame(frame_shape, onsets_indices)
    velocities_frame = tf.scatter_nd(onsets_indices, velocities, frame_shape)

    offsets_indices = tf.stack([offsets, pitches], axis=1)
    offsets_frame = _create_pianoroll__create_frame(frame_shape, offsets_indices)

    return {
        "actives": tf.clip_by_value(actives_frame, 0, 1),
        "onsets": tf.clip_by_value(onsets_frame, 0, 1),
        "offsets": tf.clip_by_value(offsets_frame, 0, 1),
        "velocities": tf.clip_by_value(velocities_frame, 0, 1),
    }


def _preprocess_sample(sample):
    spec = sample["spec"]
    num_frames = tf.shape(spec)[0]
    pianoroll = _create_pianoroll(sample["notesequence"], num_frames)
    del pianoroll["velocities"]
    return spec, pianoroll


def _split_sample__reshape_sequence(sequence):
    shape = tf.shape(sequence)
    smallest = shape[0] % MAX_LENGTH
    new_sequence = tf.cond(
        pred=smallest < MIN_LENGTH,
        true_fn=lambda: sequence[: shape[0] - smallest],
        false_fn=lambda: tf.pad(sequence, [[0, MAX_LENGTH - smallest], [0, 0]]),
    )
    return tf.reshape(new_sequence, [-1, MAX_LENGTH, shape[1]])


def _split_sample(spec, pianoroll):
    spec = _split_sample__reshape_sequence(spec)
    spec.set_shape([None, MAX_LENGTH, constants.SPEC_N_BINS])
    pianoroll = {
        key: _split_sample__reshape_sequence(value) for key, value in pianoroll.items()
    }
    for value in pianoroll.values():
        value.set_shape(
            [None, MAX_LENGTH, constants.MAX_PITCH - constants.MIN_PITCH + 1]
        )
    return tf.data.Dataset.from_tensor_slices((spec, pianoroll))


def load_dataset(path, split, num_replicas):
    ordered = False if split == "train" else True
    dataset = load_transcription_dataset(path, split, ordered)
    dataset = dataset.map(_preprocess_sample, tf.data.experimental.AUTOTUNE)
    dataset = dataset.flat_map(_split_sample)
    if not ordered:
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE_PER_REPLICA * num_replicas)
    dataset = dataset.batch(BATCH_SIZE_PER_REPLICA * num_replicas, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
