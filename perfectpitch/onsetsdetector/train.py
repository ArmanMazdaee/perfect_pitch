from collections import defaultdict

import torch
from tqdm import tqdm

from perfectpitch.utils.dataloader import padded_collate
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


def _train_epoch(loader, model, optimizer, scheduler, num_steps, device):
    results = defaultdict(list)
    model.train()
    for specs, labels, weights in tqdm(loader, desc="training", total=num_steps):
        specs = specs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        predictions = model(specs)
        result = _evaluate(predictions, labels, weights)

        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()
        scheduler.step()

        for key, value in result.items():
            results[key].append(value.detach())

    return {key: torch.stack(value).mean().item() for key, value in results.items()}


def _validate_epoch(validation_iterator, model, num_steps, device):
    results = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for specs, labels, weights in tqdm(
            validation_iterator, desc="validating", total=num_steps
        ):
            specs = specs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            predictions = model(specs)
            result = _evaluate(predictions, labels, weights)

            for key, value in result.items():
                results[key].append(value)

    return {key: torch.stack(values).mean().item() for key, values in results.items()}


def _log_results(epoch, train_result, validation_result):
    keys = set(train_result.keys())
    keys.update(validation_result.keys())
    keys = sorted(keys)

    print(f"epoch {epoch} result:")
    print("{: >20} {: >20} {: >20}".format("name", "train", "validation"))
    for key in keys:
        train = train_result.get(key, "NONE")
        validation = validation_result.get(key, "NONE")
        print("{: >20} {: >20} {: >20}".format(key, train, validation))


def train_onsets_detector(
    train_dataset_path, validation_dataset_path, model_dir, device
):
    num_epochs = 20
    device = torch.device(device)

    train_dataset = OnsetsDataset(
        train_dataset_path, shuffle=True, min_length=500, max_length=2000
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        collate_fn=padded_collate,
        num_workers=1,
        drop_last=True,
    )
    num_train_steps = sum(1 for _ in train_loader)
    validation_dataset = OnsetsDataset(validation_dataset_path, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=1, num_workers=1, drop_last=False,
    )
    num_validation_steps = sum(1 for _ in validation_loader)

    model = OnsetsDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.003, steps_per_epoch=num_train_steps, epochs=num_epochs
    )

    for epoch in range(1, num_epochs + 1):
        train_result = _train_epoch(
            train_loader, model, optimizer, scheduler, num_train_steps, device
        )
        validation_result = _validate_epoch(
            validation_loader, model, num_validation_steps, device
        )
        _log_results(epoch, train_result, validation_result)
