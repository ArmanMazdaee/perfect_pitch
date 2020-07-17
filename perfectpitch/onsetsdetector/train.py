import functools

import torch

from perfectpitch.dataset.pianoroll_dataset import PianorollDataset
from perfectpitch.utils.dataloader import padded_collate
from perfectpitch.utils.train import train_model
from .model import OnsetsDetector


def _evaluate_batch(onsets_detector, batch, device):
    spec = batch["spec"].to(device)
    posenc = batch["posenc"].to(device)
    mask = batch["mask"].to(device)
    prediction = onsets_detector(spec, posenc, mask)
    prediction = prediction[mask]
    label = batch["pianoroll"]["onsets"].to(device)
    label = label[mask]

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction, label, pos_weight=torch.tensor(2.0)
    )

    positive = prediction >= 0
    true = label >= 0.5
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

    train_dataset = PianorollDataset(
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
    validation_dataset = PianorollDataset(
        validation_dataset_path, shuffle=False, min_length=500, max_length=2000
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=8,
        num_workers=1,
        collate_fn=padded_collate,
        pin_memory=True,
        drop_last=False,
    )

    onsets_detector = OnsetsDetector().to(device)
    optimizer = torch.optim.Adam(onsets_detector.parameters())
    num_steps = sum(1 for _ in train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.003, steps_per_epoch=num_steps, epochs=num_epochs
    )

    train_model(
        train_loader,
        validation_loader,
        onsets_detector,
        functools.partial(_evaluate_batch, device=device),
        optimizer,
        scheduler,
        num_epochs,
        model_dir,
    )
