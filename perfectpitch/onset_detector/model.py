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
        inputs_shape = tf.shape(inputs)
        mask = tf.expand_dims(tf.range(inputs_shape[1]), axis=0) >= lengths
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
        spec = inputs["spec"]
        length = inputs["length"]
        x = self.embedding(spec, training=training)
        x = self.sequential(x, length, training=training)
        x = self.linear(x)
        return x
