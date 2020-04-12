import tensorflow as tf

from perfectpitch import constants


def _conv2d_stack(x):
    x = tf.keras.layers.Reshape(target_shape=(-1, x.shape[2], 1))(x)

    x = tf.keras.layers.Conv2D(
        filters=48, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Conv2D(
        filters=48, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same")(x)
    x = tf.keras.layers.Dropout(rate=0.25)(x)

    x = tf.keras.layers.Conv2D(
        filters=96, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same")(x)
    x = tf.keras.layers.Dropout(rate=0.25)(x)

    x = tf.keras.layers.Reshape(target_shape=(-1, x.shape[2] * x.shape[3]))(x)
    x = tf.keras.layers.Conv1D(filters=768, kernel_size=1)(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    return x


def create_model():
    num_pitches = constants.MAX_PITCH - constants.MIN_PITCH + 1

    spec = tf.keras.Input(shape=[None, constants.SPEC_N_BINS])
    onsets = _conv2d_stack(spec)
    onsets = tf.keras.layers.Conv1D(filters=num_pitches, kernel_size=1, name="onsets")(
        onsets
    )

    offsets = _conv2d_stack(spec)
    offsets = tf.keras.layers.Conv1D(
        filters=num_pitches, kernel_size=1, name="offsets"
    )(offsets)

    actives = _conv2d_stack(spec)
    actives = tf.keras.layers.Conv1D(
        filters=num_pitches, kernel_size=1, name="actives"
    )(actives)

    model = tf.keras.Model(inputs=spec, outputs=[onsets, offsets, actives])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            "onsets": tf.keras.losses.BinaryCrossentropy(
                from_logits=False, reduction=tf.keras.losses.Reduction.SUM
            ),
            "offsets": tf.keras.losses.BinaryCrossentropy(
                from_logits=False, reduction=tf.keras.losses.Reduction.SUM
            ),
            "actives": tf.keras.losses.BinaryCrossentropy(
                from_logits=False, reduction=tf.keras.losses.Reduction.SUM
            ),
        },
    )

    return model
