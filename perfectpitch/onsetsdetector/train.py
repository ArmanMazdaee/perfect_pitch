import torch

from perfectpitch.utils.dataloader import padded_collate
from perfectpitch.utils.train import train_model
from .dataset import OnsetsDataset
from .model import OnsetsDetector


def _evaluate(predictions, labels, weights):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        predictions, labels, weights
    )

    positive = torch.zeros_like(predictions)
    positive[(predictions >= 0) & (weights >= 0)] = 1
    true = torch.zeros_like(labels)
    true[(labels >= 0.5) & (weights >= 0)] = 1
    true_positive = positive * true
    precision = true_positive.sum() / (positive.sum() + 1e-6)
    recall = true_positive.sum() / (true.sum() + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {"loss": loss, "precision": precision, "recall": recall, "f1": f1}


def train_onsets_detector(
    train_dataset_path, validation_dataset_path, model_dir, device
):
    num_epochs = 40
    device = torch.device(device)

    train_dataset = OnsetsDataset(
        train_dataset_path, shuffle=True, min_length=500, max_length=2000
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=1,
        collate_fn=padded_collate,
        pin_memory=True,
        drop_last=True,
    )
    validation_dataset = OnsetsDataset(validation_dataset_path, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    model = OnsetsDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    num_train_steps = sum(1 for _ in train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.005, steps_per_epoch=num_train_steps, epochs=num_epochs
    )

    train_model(
        train_loader,
        validation_loader,
        model,
        _evaluate,
        optimizer,
        scheduler,
        num_epochs,
        device,
        model_dir,
    )
