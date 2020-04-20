import pytorch_lightning as pl

from .model import AcousticModel


def train_acoustic(train_dataset_path, validation_dataset_path, device, model_dir):
    model = AcousticModel(train_dataset_path, validation_dataset_path)
    trainer = pl.Trainer(
        default_root_dir=model_dir,
        gpus=-1 if device == "gpu" else 0,
        num_tpu_cores=8 if device == "tpu" else None,
        max_epochs=3,
    )
    trainer.fit(model)
