import tensorflow as tf


class WeightNormConv1D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        dilation_rate=1,
        activation=None,
        use_bias=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel_v = self.add_weight(
            name="kernel_v",
            shape=[self.kernel_size, input_shape[-1], self.filters],
            initializer=tf.random_normal_initializer(0, 0.01),
            trainable=True,
            dtype=self.dtype,
        )
        self.kernel_g = self.add_weight(
            name="kernel_g",
            shape=[1, 1, self.filters],
            initializer=tf.constant_initializer(1.0),
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[1, 1, self.filters],
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None

    def call(self, inputs):
        kernel = self.kernel_g * tf.keras.backend.l2_normalize(self.kernel_v)
        outputs = tf.keras.backend.conv1d(
            inputs,
            kernel,
            self.strides,
            self.padding,
            "channels_last",
            self.dilation_rate,
        )

        if self.bias is not None:
            outputs += self.bias

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "activation": self.activation,
                "use_bias": self.use_bias,
            }
        )
        return config


class TemporalConv1DNetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="same",
        dilation_rate=1,
        dropout=0.2,
        use_bias=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.use_bias = use_bias

        self.conv1 = WeightNormConv1D(
            filters, kernel_size, strides, padding, dilation_rate, "relu", use_bias
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.conv2 = WeightNormConv1D(
            filters, kernel_size, strides, padding, dilation_rate, "relu", use_bias
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.downsample = WeightNormConv1D(filters, 1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)

        if inputs.shape[-1] != x.shape[-1]:
            x += self.downsample(inputs)
        else:
            x += inputs

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "dropout": self.dropout,
                "use_bias": self.use_bias,
            }
        )
        return config


class TemporalConv1DNet(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="same",
        dilation_rate=1,
        dropout=0.2,
        use_bias=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.use_bias = use_bias

        self.blocks = [
            TemporalConv1DNetBlock(
                f, kernel_size, strides, padding, 2 ** i, dropout, use_bias
            )
            for i, f in enumerate(filters)
        ]

    def call(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "dropout": self.dropout,
                "use_bias": self.use_bias,
            }
        )
        return config
