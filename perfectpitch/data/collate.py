import torch


def _pad_to_shape(tensor, shape):
    pad = []
    for i in range(len(shape)):
        pad.append(shape[i] - tensor.shape[i])
        pad.append(0)
    pad.reverse()
    return torch.nn.functional.pad(tensor, pad)


def padded_collate(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        padded_shape = [max([b.shape[i] for b in batch]) for i in range(elem.ndim)]
        padded_batch = [_pad_to_shape(t, padded_shape) for t in batch]
        return torch.stack(padded_batch)
    elif isinstance(elem, dict):
        return {key: padded_collate([b[key] for b in batch]) for key in elem.keys()}
    raise TypeError(f"{type(elem)} is not supported")
