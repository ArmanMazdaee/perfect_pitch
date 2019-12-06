import math

import torch
import pytorch_lightning as pl

from perfectpitch import constants
from perfectpitch.data.dataset import Dataset
from perfectpitch.data.collate import padded_collate


def _binary_cross_entropy_with_logits(input, target, weight):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        input, target, reduction="none"
    )
    return (loss * weight).sum() / weight.sum()


class _Conv2dStack(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv2ds = torch.nn.Sequential(
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=1, out_channels=48, kernel_size=3, padding=1,
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=48,
                    out_channels=48,
                    kernel_size=3,
                    stride=(2, 1),
                    padding=1,
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=48,
                    out_channels=96,
                    kernel_size=3,
                    stride=(2, 1),
                    padding=1,
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.linear = torch.nn.Conv1d(
            96 * math.ceil(in_channels / 4), out_channels, kernel_size=1,
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv2ds(x)
        x = x.flatten(1, 2)
        x = self.linear(x)
        return x


class Acoustic(pl.LightningModule):
    def __init__(self, train_path=None, val_path=None):
        super().__init__()
        in_channels = constants.SPEC_N_BINS
        out_channels = constants.MAX_PITCH - constants.MIN_PITCH + 1
        dropout = 0.2
        self.onsets_stack = _Conv2dStack(in_channels, out_channels, dropout)
        self.offsets_stack = _Conv2dStack(in_channels, out_channels, dropout)
        self.actives_stack = _Conv2dStack(in_channels, out_channels, dropout)
        self.train_path = train_path
        self.val_path = val_path

    def forward(self, spec):
        pianoroll = {}
        pianoroll["onsets"] = self.onsets_stack(spec)
        pianoroll["offsets"] = self.offsets_stack(spec)
        pianoroll["actives"] = self.actives_stack(spec)
        return pianoroll

    def loss(self, prediction, label, weight):
        onsets_loss = _binary_cross_entropy_with_logits(
            prediction["onsets"], label["onset"], weight["onset"],
        )
        offsets_loss = _binary_cross_entropy_with_logits(
            prediction["offsets"], label["offsets"], weight["offsets"],
        )
        actives_loss = _binary_cross_entropy_with_logits(
            prediction["actives"], label["actives"], weight["actives"],
        )
        total_loss = (onsets_loss + offsets_loss + actives_loss) / 3
        return {
            "onsets_loss": onsets_loss,
            "offsets_loss": offsets_loss,
            "actives_loss": actives_loss,
            "total_loss": total_loss,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0006)

    @pl.data_loader
    def train_dataloader(self):
        if self.train_path is None:
            return None
        trainset = Dataset(
            self.train_path, spec=True, pianoroll=True, pianoroll_weight=True
        )
        return torch.utils.data.DataLoader(
            trainset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            collate_fn=padded_collate,
            drop_last=True,
        )

    @pl.data_loader
    def val_dataloader(self):
        if self.val_path is None:
            return None
        valset = Dataset(
            self.val_path, spec=True, pianoroll=True, pianoroll_weight=True
        )
        return torch.utils.data.DataLoader(
            valset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            collate_fn=padded_collate,
            drop_last=True,
        )

    def training_step(self, batch, batch_nb):
        spec = batch["spec"]
        pianoroll = batch["pianoroll"]
        pianoroll_weight = batch["pianoroll_weight"]
        prediction = self.forward(spec)
        loss = self.loss(prediction, pianoroll, pianoroll_weight)
        return {"loss": loss["total_loss"], "log": loss}

    def validation_step(self, batch, batch_nb):
        spec = batch["spec"]
        pianoroll = batch["pianoroll"]
        pianoroll_weight = batch["pianoroll_weight"]
        prediction = self.forward(spec)
        loss = self.loss(prediction, pianoroll, pianoroll_weight)
        return loss
