from collections import defaultdict

import torch
from tqdm import tqdm

from .dataset import OnsetsDataset
from .model import OnsetsDetector


def _evaluate_prediction(prediction, label):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction, label, pos_weight=torch.tensor(5.0)
    )

    positive = torch.zeros_like(prediction)
    positive[prediction >= 0] = 1
    true = torch.zeros_like(label)
    true[label >= 0.5] = 1
    true_positive = positive * true
    precision = true_positive.sum() / (positive.sum() + 1e-6)
    recall = true_positive.sum() / (true.sum() + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {"loss": loss, "precision": precision, "recall": recall, "f1": f1}


def _train_epoch(loader, model, optimizer, device):
    results = defaultdict(list)
    model.train()
    for spec, label in tqdm(loader, desc="training"):
        spec = spec.to(device)
        label = label.to(device)
        prediction = model(spec)
        result = _evaluate_prediction(prediction, label)

        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()

        for key, value in result.items():
            results[key].append(value.detach())

    return {key: torch.stack(value).mean().item() for key, value in results.items()}


def _validate_epoch(validation_iterator, model, device):
    results = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for spec, label in tqdm(validation_iterator, desc="validating"):
            spec = spec.to(device)
            label = label.to(device)
            prediction = model(spec)
            result = _evaluate_prediction(prediction, label)

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
    num_epochs = 10
    device = torch.device(device)
    train_dataset = OnsetsDataset(train_dataset_path, min_length=150, max_length=4000)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )
    validation_dataset = OnsetsDataset(validation_dataset_path)
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )
    model = OnsetsDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(1, num_epochs + 1):
        train_result = _train_epoch(train_loader, model, optimizer, device)
        validation_result = _validate_epoch(validation_loader, model, device)
        _log_results(epoch, train_result, validation_result)
