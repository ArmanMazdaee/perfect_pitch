import tensorflow as tf

from perfectpitch import constants
from perfectpitch.utils.layers import PositionalEncoding, SelfAttention


class _Sequential(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, d_model, d_feedforward, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.selfattentions = [
            SelfAttention(d_model=d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ]
        self.dropout1s = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]
        self.layernorm1s = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)
        ]

        self.feedforwards = [
            tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(d_feedforward, activation="relu"),
                    tf.keras.layers.Dense(d_model),
                ]
            )
            for _ in range(num_layers)
        ]
        self.dropout2s = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]
        self.layernorm2s = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)
        ]

    def call(self, inputs, lengths, training):
        max_length = tf.shape(inputs)[1]
        mask = tf.sequence_mask(lengths, max_length, dtype=tf.dtypes.float32)
        mask = 1 - mask
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.cast(mask, inputs.dtype)

        x = inputs
        for i in range(self.num_layers):
            y = self.selfattentions[i](x, mask)
            y = self.dropout1s[i](y, training)
            x = self.layernorm1s[i](x + y)

            y = self.feedforwards[i](x)
            y = self.dropout2s[i](y, training)
            x = self.layernorm2s[i](x + y)

        return x


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512),
                PositionalEncoding(),
                tf.keras.layers.Dropout(0.1),
            ]
        )
        self.sequential = _Sequential(
            num_layers=8, num_heads=4, d_model=512, d_feedforward=2048, dropout=0.1
        )
        self.linear = tf.keras.layers.Dense(constants.NUM_PITCHES)

    def call(self, inputs, training=False):
        spec, length = inputs
        x = self.embedding(spec, training=training)
        x = self.sequential(x, length, training=training)
        x = self.linear(x)
        return x

    def train_step(self, data):
        spec = data["spec"]
        length = data["length"]

        max_length = tf.shape(spec)[1]
        mask = tf.sequence_mask(length, max_length)
        labels = tf.boolean_mask(data["onsets"], mask)
        with tf.GradientTape() as tape:
            predictions = tf.boolean_mask(self((spec, length), training=True), mask)
            loss = self.compiled_loss(
                labels, predictions, regularization_losses=self.losses
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(labels, tf.nn.sigmoid(predictions))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        spec = data["spec"]
        length = data["length"]

        max_length = tf.shape(spec)[1]
        mask = tf.sequence_mask(length, max_length)
        labels = tf.boolean_mask(data["onsets"], mask)
        predictions = tf.boolean_mask(self((spec, length), training=False), mask)

        self.compiled_loss(labels, predictions, regularization_losses=self.losses)
        self.compiled_metrics.update_state(labels, tf.nn.sigmoid(predictions))
        return {m.name: m.result() for m in self.metrics}
