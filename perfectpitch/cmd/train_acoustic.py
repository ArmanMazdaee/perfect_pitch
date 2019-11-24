import collections

from tqdm import tqdm
import torch

from perfectpitch.data.dataset import Dataset
from perfectpitch.data.collate import padded_collate
from perfectpitch.models.acoustic import Acoustic


def _get_dataset(path, shuffle):
    dataset = Dataset(path, spec=True, pianoroll=True, pianoroll_weight=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=padded_collate,
        drop_last=True,
    )
    return dataloader


def _binary_cross_entropy_with_logits(input, target, weight):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        input, target, reduction="none"
    )
    return (loss * weight).sum() / weight.sum()


def _batch_criterion(data, model):
    spec = data["spec"].to(model.device)
    pianoroll = data["pianoroll"].to(model.device)
    pianoroll_weight = data["pianoroll_weight"].to(model.device)

    prediction = model(spec)
    onsets_loss = _binary_cross_entropy_with_logits(
        prediction["onsets"], pianoroll["onsets"], pianoroll_weight["onsets"]
    )
    total_loss = onsets_loss
    return {
        "onsets_loss": onsets_loss,
        "total_loss": total_loss,
    }


def train_acoustic(train_path, validation_path, use_gpu):
    train_loader = _get_dataset(train_path, shuffle=True)
    validation_loader = _get_dataset(validation_path, shuffle=False)

    model = Acoustic()
    if use_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.98)

    for epoch in range(1, 2):
        model.train()
        train_loss = collections.defaultdict(lambda: 0.0)
        for data in tqdm(
            train_loader, desc=f"training epoch {epoch}", position=0, leave=True
        ):
            optimizer.zero_grad()
            loss = _batch_criterion(data, model)
            loss["total_loss"].backward()
            optimizer.step()
            scheduler.step()
            for key in loss.keys():
                train_loss[key] += loss[key].item()

        model.eval()
        validation_loss = collections.defaultdict(lambda: 0.0)
        with torch.no_grad():
            for data in tqdm(
                validation_loader,
                desc=f"validating epoch {epoch}",
                position=0,
                leave=True,
            ):
                loss = _batch_criterion(data, model)
                for key in loss.keys():
                    validation_loss[key] += loss[key].item()

        print("epoch:", epoch)
        for key, value in train_loss.items():
            print(key + "/train", value / len(train_loader))
        for key, value in validation_loss.items():
            print(key + "/train", value / len(validation_loader))
        print()