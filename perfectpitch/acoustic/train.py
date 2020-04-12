import tensorflow as tf

from .dataset import load_dataset
from .model import create_model


def train(dataset_path, distribute_strategy):
    if distribute_strategy == "single":
        strategy = tf.distribute.OneDeviceStrategy(
            "/gpu:0" if tf.test.is_gpu_available() else "/cpu"
        )
    elif distribute_strategy == "tpu":
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        raise ValueError(f"distribute_strategy {distribute_strategy} is unknown")

    with strategy.scope():
        train_data = load_dataset(
            dataset_path, "train", strategy.num_replicas_in_sync
        ).take(2)
        validation_data = load_dataset(
            dataset_path, "validation", strategy.num_replicas_in_sync
        ).take(2)
        model = create_model()
        model.fit(
            x=train_data, epochs=2, validation_data=validation_data,
        )
