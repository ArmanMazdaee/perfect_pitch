import torch
import pytorch_lightning as pl

from perfectpitch import constants
from .dataset import AcousticDataset


class _Conv2dStack(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=48, kernel_size=3, padding=1
        )
        self.activation1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(
            in_channels=48, out_channels=48, kernel_size=3, padding=1
        )
        self.activation2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout2 = torch.nn.Dropout(0.25)

        self.conv3 = torch.nn.Conv2d(
            in_channels=48, out_channels=96, kernel_size=3, padding=1
        )
        self.activation3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout3 = torch.nn.Dropout(0.25)

        self.linear = torch.nn.Conv1d(
            in_channels=96 * (constants.SPEC_N_BINS // 4),
            out_channels=768,
            kernel_size=1,
        )
        self.activation_linear = torch.nn.ReLU()
        self.dropout_linear = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.activation3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)

        x = self.linear(x)
        x = self.activation_linear(x)
        x = self.dropout_linear(x)
        return x


class AcousticModel(pl.LightningModule):
    def __init__(self, train_dataset_path=None, validation_dataset_path=None):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.validation_dataset_path = validation_dataset_path

        self.onsets_conv2d_stack = _Conv2dStack()
        self.onsets_linear = torch.nn.Conv1d(
            in_channels=768, out_channels=88, kernel_size=1
        )
        self.offsets_conv2d_stack = _Conv2dStack()
        self.offsets_linear = torch.nn.Conv1d(
            in_channels=768, out_channels=88, kernel_size=1
        )
        self.actives_conv2d_stack = _Conv2dStack()
        self.actives_linear = torch.nn.Conv1d(
            in_channels=768, out_channels=88, kernel_size=1
        )

    def forward(self, spec):
        onsets = self.onsets_conv2d_stack(spec)
        onsets = self.onsets_linear(onsets)
        offsets = self.offsets_conv2d_stack(spec)
        offsets = self.offsets_linear(offsets)
        actives = self.actives_conv2d_stack(spec)
        actives = self.actives_linear(actives)
        return {"onsets": onsets, "offsets": offsets, "actives": actives}

    @property
    def train_dataset_path(self):
        return self.__train_dataset_path

    @train_dataset_path.setter
    def train_dataset_path(self, path):
        self.__train_dataset_path = path
        self.__train_dataset = None

    @property
    def validation_dataset_path(self):
        return self.__validation_dataset_path

    @validation_dataset_path.setter
    def validation_dataset_path(self, path):
        self.__validation_dataset_path = path
        self.__validation_dataset = None

    def prepare_data(self):
        if not hasattr(self, "__train_dataset") or self.__train_dataset is None:
            self.__train_dataset = AcousticDataset(
                self.train_dataset_path,
                min_length=100,
                max_length=625,
                pad_sequences=True,
            )

        if not hasattr(self, "__validation_dataset") or self.__train_dataset is None:
            self.__validation_dataset = AcousticDataset(
                self.validation_dataset_path,
                min_length=100,
                max_length=625,
                pad_sequences=True,
            )

    def evaluate_prediction(self, perdiction, label):
        onsets_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            perdiction["onsets"], label["onsets"]
        )
        offsets_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            perdiction["offsets"], label["offsets"]
        )
        actives_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            perdiction["actives"], label["actives"]
        )
        loss = onsets_loss + offsets_loss + actives_loss

        return {
            "onsets_loss": onsets_loss,
            "offsets_loss": offsets_loss,
            "actives_loss": actives_loss,
            "loss": loss,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.__train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

    def training_step(self, batch, batch_idx):
        spec = batch["spec"]
        prediction = self(spec)
        label = batch["pianoroll"]
        results = self.evaluate_prediction(prediction, label)
        return {
            "loss": results["loss"],
            "log": results,
        }

    def training_epoch_end(self, outputs):
        log = {
            key: torch.stack([output["log"][key] for output in outputs]).mean()
            for key in outputs[0]["log"].keys()
        }
        return {"log": log}

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.__validation_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            drop_last=True,
        )

    def validation_step(self, batch, batch_idx):
        spec = batch["spec"]
        prediction = self(spec)
        label = batch["pianoroll"]
        results = self.evaluate_prediction(prediction, label)
        return {
            "log": results,
        }

    def validation_epoch_end(self, outputs):
        log = {
            key: torch.stack([output["log"][key] for output in outputs]).mean()
            for key in outputs[0]["log"].keys()
        }
        return {"log": log}
