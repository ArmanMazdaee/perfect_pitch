import os
from collections import defaultdict

import torch
from tqdm import tqdm


def _train_epoch(train_loader, model, evaluate, optimizer, scheduler, epoch, device):
    results = defaultdict(list)
    model.train()
    for inputs, labels, weights in tqdm(train_loader, desc=f"training epoch {epoch}"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        predictions = model(inputs)
        result = evaluate(predictions, labels, weights)

        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()
        scheduler.step()

        for key, value in result.items():
            results[key].append(value.detach())

    return {key: torch.stack(value).mean().item() for key, value in results.items()}


def _validate_epoch(validation_loader, model, evaluate, epoch, device):
    results = defaultdict(list)
    model.eval()
    for inputs, labels, weights in tqdm(
        validation_loader, desc=f"validating epoch {epoch}"
    ):
        inputs = inputs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        with torch.no_grad():
            predictions = model(inputs)
            result = evaluate(predictions, labels, weights)

        for key, value in result.items():
            results[key].append(value.detach())

    return {key: torch.stack(value).mean().item() for key, value in results.items()}


def _print_result(train_result, validation_result, epoch):
    keys = set(train_result.keys())
    keys.update(validation_result.keys())
    keys = sorted(keys)

    print(f"epoch {epoch} result:")
    print("{: >20} {: >20} {: >20}".format("name", "train", "validation"))
    for key in keys:
        train = train_result.get(key, "NONE")
        validation = validation_result.get(key, "NONE")
        print("{: >20} {: >20} {: >20}".format(key, train, validation))


def _save_model(model, epoch, model_dir):
    state_dict = model.state_dict()
    path = os.path.join(model_dir, f"weights-epoch-{epoch}.pt")
    torch.save(state_dict, path)


def train_model(
    train_loader,
    validation_loader,
    model,
    evaluate,
    optimizer,
    scheduler,
    num_epochs,
    device,
    model_dir,
):
    os.makedirs(model_dir, exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        train_result = _train_epoch(
            train_loader, model, evaluate, optimizer, scheduler, epoch, device
        )
        validation_result = _validate_epoch(
            validation_loader, model, evaluate, epoch, device
        )
        _print_result(train_result, validation_result, epoch)
        _save_model(model, epoch, model_dir)
