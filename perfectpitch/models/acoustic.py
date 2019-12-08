import math

import torch

from perfectpitch import constants
from perfectpitch.data.dataset import Dataset
from perfectpitch.data.collate import padded_collate


def _binary_cross_entropy_with_logits(input, target, weight):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        input, target, reduction="none"
    )
    return (loss * weight).sum() / weight.sum()


class _Conv2dStack(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv2ds = torch.nn.Sequential(
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=1, out_channels=48, kernel_size=3, padding=1,
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=48,
                    out_channels=48,
                    kernel_size=3,
                    stride=(2, 1),
                    padding=1,
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=48,
                    out_channels=96,
                    kernel_size=3,
                    stride=(2, 1),
                    padding=1,
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.linear = torch.nn.Conv1d(
            96 * math.ceil(in_channels / 4), out_channels, kernel_size=1,
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv2ds(x)
        x = x.flatten(1, 2)
        x = self.linear(x)
        return x


class Acoustic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = constants.SPEC_N_BINS
        out_channels = constants.MAX_PITCH - constants.MIN_PITCH + 1
        dropout = 0.2
        self.onsets_stack = _Conv2dStack(in_channels, out_channels, dropout)
        self.offsets_stack = _Conv2dStack(in_channels, out_channels, dropout)
        self.actives_stack = _Conv2dStack(in_channels, out_channels, dropout)

    def forward(self, spec):
        pianoroll = {}
        pianoroll["onsets"] = self.onsets_stack(spec)
        pianoroll["offsets"] = self.offsets_stack(spec)
        pianoroll["actives"] = self.actives_stack(spec)
        return pianoroll
