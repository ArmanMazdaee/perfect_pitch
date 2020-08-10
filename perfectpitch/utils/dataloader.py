import numpy as np
import torch


def padded_collate(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    elif isinstance(elem, np.ndarray):
        return padded_collate([torch.from_numpy(t) for t in batch])
    elif type(elem) == dict:
        return {key: padded_collate([b[key] for b in batch]) for key in elem.keys()}
    elif type(elem) == tuple:
        return tuple(
            padded_collate([b[index] for b in batch]) for index, _ in enumerate(elem)
        )
    raise TypeError(f"{type(elem)} is not supported")
