import math

import torch

from perfectpitch import constants
from perfectpitch.modules.tcn import TemporalConvNet


class _Conv2dStack(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv2ds = torch.nn.Sequential(
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=out_channels // 16,
                    kernel_size=3,
                    padding=1,
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=out_channels // 16,
                    out_channels=out_channels // 16,
                    kernel_size=3,
                    stride=(2, 1),
                    padding=1,
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=out_channels // 16,
                    out_channels=out_channels // 8,
                    kernel_size=3,
                    stride=(2, 1),
                    padding=1,
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Conv1d(
                (out_channels // 8) * math.ceil(in_channels / 4),
                out_channels,
                kernel_size=1,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv2ds(x)
        x = x.flatten(1, 2)
        x = self.linear(x)
        return x


class _SequentialStack(TemporalConvNet):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__(
            in_channels, out_channels, [128] * 6, 3, "same", dropout,
        )


class Acoustic(torch.nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = constants.SPEC_N_BINS
        middle_channels = 768
        out_channels = constants.MAX_PITCH - constants.MIN_PITCH + 1
        dropout = 0.2

        self.onsets_stack = torch.nn.Sequential(
            _Conv2dStack(in_channels, middle_channels, dropout),
            _SequentialStack(middle_channels, out_channels, dropout),
        )

    def forward(self, spec):
        pianoroll = {}

        onsets = self.onsets_stack(spec)
        pianoroll["onsets"] = onsets

        return pianoroll
