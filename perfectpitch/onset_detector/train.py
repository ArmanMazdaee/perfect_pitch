import tensorflow as tf

from .dataset import get_dataset
from .model import Model


def train(train_dataset_path, validation_dataset_path):
    train_dataset = get_dataset(
        train_dataset_path, batch_size=16, shuffle=True, min_length=150, max_length=1000
    )
    validation_dataset = get_dataset(
        validation_dataset_path,
        batch_size=16,
        shuffle=False,
        min_length=150,
        max_length=1000,
    )

    model = Model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    model.fit(x=train_dataset, validation_data=validation_dataset, epochs=10)
