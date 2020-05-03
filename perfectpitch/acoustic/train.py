from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import AcousticDataset
from .model import AcousticModel


def _batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {key: _batch_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)


def _evaluate_prediction(perdiction, label):
    onsets_loss = F.binary_cross_entropy_with_logits(
        perdiction["onsets"], label["onsets"]
    )
    offsets_loss = F.binary_cross_entropy_with_logits(
        perdiction["offsets"], label["offsets"]
    )
    actives_loss = F.binary_cross_entropy_with_logits(
        perdiction["actives"], label["actives"]
    )
    loss = onsets_loss + offsets_loss + actives_loss

    return {
        "onsets_loss": onsets_loss,
        "offsets_loss": offsets_loss,
        "actives_loss": actives_loss,
        "loss": loss,
    }


def _train_epoch(loader, model, optimizer, device):
    results = defaultdict(list)
    model.train()
    for batch in tqdm(loader, desc="training"):
        batch = _batch_to_device(batch, device)
        spec = batch["spec"]
        label = batch["pianoroll"]
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
        for batch in tqdm(validation_iterator, desc="validating"):
            batch = _batch_to_device(batch, device)
            spec = batch["spec"]
            label = batch["pianoroll"]
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


def train_acoustic(train_dataset_path, validation_dataset_path, model_dir, device):
    device = torch.device(device)
    train_dataset = AcousticDataset(train_dataset_path, min_length=150, max_length=4000)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )
    validation_dataset = AcousticDataset(validation_dataset_path)
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )
    model = AcousticModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 6):
        train_result = _train_epoch(train_loader, model, optimizer, device)
        validation_result = _validate_epoch(validation_loader, model, device)
        _log_results(epoch, train_result, validation_result)
