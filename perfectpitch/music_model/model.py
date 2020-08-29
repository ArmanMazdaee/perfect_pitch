import tensorflow as tf

from perfectpitch import constants
from perfectpitch.utils.layers import PositionalEncoding, SelfAttention


class _Sequential(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, d_model, d_feedforward, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.positional_encoding = PositionalEncoding()

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

        self.onsets = tf.keras.layers.Dense(
            units=constants.NUM_PITCHES, activation="sigmoid",
        )
        self.offsets = tf.keras.layers.Dense(
            units=constants.NUM_PITCHES, activation="sigmoid",
        )

    def call(self, inputs, lengths, training):
        max_length = tf.shape(inputs)[1]
        mask = tf.sequence_mask(lengths, max_length, dtype=tf.dtypes.float32)
        mask = 1 - mask
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.cast(mask, inputs.dtype)

        x = self.positional_encoding(inputs)
        for i in range(self.num_layers):
            y = self.selfattentions[i](x, mask)
            y = self.dropout1s[i](y, training)
            x = self.layernorm1s[i](x + y)

            y = self.feedforwards[i](x)
            y = self.dropout2s[i](y, training)
            x = self.layernorm2s[i](x + y)

        return self.onsets(x), self.offsets(x)


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
            ]
        )
        self.sequential = _Sequential(
            num_layers=8, num_heads=4, d_model=512, d_feedforward=2048, dropout=0.1
        )

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss={
                "onsets": tf.keras.losses.BinaryCrossentropy(),
                "offsets": tf.keras.losses.BinaryCrossentropy(),
            },
            metrics={
                "onsets": [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                "offsets": [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            },
        )

    def call(self, inputs, training=False):
        embedding_input = tf.concat([inputs["onsets"], inputs["offsets"]], axis=-1)
        embedding_output = self.embedding(embedding_input, training=training)
        if "mask" in inputs:
            mask = 1 - tf.cast(inputs["mask"], embedding_output.dtype)
            embedding_output = embedding_output * tf.expand_dims(mask, axis=-1)

        onsets, offsets = self.sequential(
            embedding_output, inputs["length"], training=training
        )
        return {
            "onsets": onsets,
            "offsets": offsets,
        }

    def train_step(self, data):
        labels = {
            key: tf.boolean_mask(data[key], data["mask"])
            for key in ["onsets", "offsets"]
        }
        with tf.GradientTape() as tape:
            predictions = self(data, training=True)
            predictions = {
                key: tf.boolean_mask(predictions[key], data["mask"])
                for key in ["onsets", "offsets"]
            }
            loss = self.compiled_loss(
                labels, predictions, regularization_losses=self.losses
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labels = {
            key: tf.boolean_mask(data[key], data["mask"])
            for key in ["onsets", "offsets"]
        }
        predictions = self(data, training=False)
        predictions = {
            key: tf.boolean_mask(predictions[key], data["mask"])
            for key in ["onsets", "offsets"]
        }
        self.compiled_loss(labels, predictions, regularization_losses=self.losses)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}
