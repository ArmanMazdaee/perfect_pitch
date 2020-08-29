import os

import tensorflow as tf

from .dataset import get_dataset
from .model import Model


def train(train_dataset_path, validation_dataset_path, model_dir):
    train_dataset = get_dataset(
        train_dataset_path,
        batch_size=16,
        shuffle=True,
        augment=False,
        min_length=150,
        max_length=1000,
    )
    validation_dataset = get_dataset(
        validation_dataset_path,
        batch_size=16,
        shuffle=False,
        augment=False,
        min_length=150,
        max_length=1000,
    )
    model = Model()

    os.makedirs(model_dir, exist_ok=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "checkpoints", "epoch-{epoch:02d}"),
        save_weights_only=True,
    )
    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        filename=os.path.join(model_dir, "logs.csv"),
    )
    model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=[model_checkpoint_callback, csv_logger_callback],
    )
