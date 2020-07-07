import numpy as np
import torch


def padded_collate(batch, batch_first=False, padding_value=0):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first, padding_value)
    elif isinstance(elem, np.ndarray):
        return padded_collate(
            [torch.from_numpy(t) for t in batch], batch_first, padding_value
        )
    elif type(elem) == dict:
        return {
            key: padded_collate([b[key] for b in batch], batch_first, padding_value)
            for key in elem.keys()
        }
    elif type(elem) == tuple:
        return tuple(
            padded_collate([b[index] for b in batch], batch_first, padding_value)
            for index, _ in enumerate(elem)
        )
    raise TypeError(f"{type(elem)} is not supported")
