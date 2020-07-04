import os
from collections import defaultdict

import torch
from tqdm import tqdm


def _train_epoch(loader, model, evaluate_batch, optimizer, scheduler):
    results = defaultdict(list)
    model.train()
    for batch in tqdm(loader, desc="training"):
        result = evaluate_batch(model, batch)

        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()
        scheduler.step()

        for key, value in result.items():
            results[key].append(value.detach())

    return {key: torch.stack(value).mean().item() for key, value in results.items()}


def _validate_epoch(loader, model, evaluate_batch):
    results = defaultdict(list)
    model.eval()
    for batch in tqdm(loader, desc="validating"):
        with torch.no_grad():
            result = evaluate_batch(model, batch)

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


def train_model(
    train_loader,
    validation_loader,
    model,
    evaluate_batch,
    optimizer,
    scheduler,
    num_epochs,
    model_dir,
):
    os.makedirs(model_dir, exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        train_result = _train_epoch(
            train_loader, model, evaluate_batch, optimizer, scheduler
        )
        validation_result = _validate_epoch(validation_loader, model, evaluate_batch)
        torch.save(
            model.state_dict(), os.path.join(model_dir, f"weights-epoch-{epoch}.pt")
        )
        _print_result(train_result, validation_result, epoch)
