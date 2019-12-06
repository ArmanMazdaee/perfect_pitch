import collections

import pytorch_lightning as pl

from perfectpitch.models.acoustic import Acoustic


def train_acoustic(train_path, validation_path, use_gpu):
    model = Acoustic(train_path, validation_path)
    trainer = pl.Trainer()
    trainer.fit(model)
