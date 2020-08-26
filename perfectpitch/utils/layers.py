import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        length = tf.cast(input_shape[1], inputs.dtype)
        dimension = tf.cast(input_shape[2], inputs.dtype)
        position = tf.range(length, delta=1)
        position = tf.expand_dims(position, axis=1)
        div_term = tf.range(dimension, delta=2)
        div_term = tf.math.exp(div_term * -tf.math.log(10000.0) / dimension)
        points = position * div_term
        sin = tf.math.sin(points)
        cos = tf.math.cos(points)
        encoding = tf.concat([sin, cos], axis=1)
        return inputs + encoding


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        if d_model % num_heads != 0:
            raise ValueError("d_model should be divisible by num_heads")

        super().__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

    def call(self, x, mask):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        length = input_shape[1]

        q = self.wq(x)
        q = tf.reshape(q, (batch_size, length, self.num_heads, self.depth))
        q = tf.transpose(q, (2, 0, 1, 3))
        k = self.wk(x)
        k = tf.reshape(k, (batch_size, length, self.num_heads, self.depth))
        k = tf.transpose(k, (2, 0, 1, 3))
        v = self.wv(x)
        v = tf.reshape(v, (batch_size, length, self.num_heads, self.depth))
        v = tf.transpose(v, (2, 0, 1, 3))

        qk = tf.linalg.matmul(q, k, transpose_b=True)
        scale = tf.math.sqrt(tf.cast(self.depth, dtype=qk.dtype))
        weights_logits = (qk / scale) + (mask * -1e9)
        weights = tf.nn.softmax(weights_logits, axis=-1)
        attention = tf.linalg.matmul(weights, v)

        result = tf.transpose(attention, (1, 2, 0, 3))
        result = tf.reshape(result, (batch_size, length, self.num_heads * self.depth))
        return result
