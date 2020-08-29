import tensorflow as tf


class BinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(
        self, from_logits=False, pos_weight=1.0, **kwargs,
    ):
        super().__init__()
        self.from_logits = from_logits
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        loss = tf.keras.backend.binary_crossentropy(y_true, y_pred, self.from_logits)
        weight = y_true * (self.pos_weight - 1) + 1
        weighted_loss = loss * weight
        return tf.math.reduce_mean(weighted_loss, axis=-1)
