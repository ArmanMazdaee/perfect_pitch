import math

import torch

from perfectpitch.modules.utils import Pad


class TemporalConvNetBlock(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, dilation, dropout
    ):
        super().__init__()
        total_pad = (kernel_size - 1) * dilation
        if padding == "same":
            padding_value = math.ceil(total_pad / 2), math.floor(total_pad / 2)
        elif padding == "causal":
            padding_value = total_pad, 0
        else:
            raise ValueError("padding should be same or causal")

        self.convs = torch.nn.Sequential(
            Pad(padding_value),
            torch.nn.utils.weight_norm(
                torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size, dilation=dilation
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Pad(padding_value),
            torch.nn.utils.weight_norm(
                torch.nn.Conv1d(
                    out_channels, out_channels, kernel_size, dilation=dilation
                )
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

        if in_channels != out_channels:
            self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = None

    def forward(self, inputs):
        x = self.convs(inputs)

        if self.downsample is not None:
            x += self.downsample(inputs)
        else:
            x += inputs

        return x


class TemporalConvNet(torch.nn.Sequential):
    def __init__(
        self, in_channels, out_channels, conv_channels, kernel_size, padding, dropout
    ):
        channels = [in_channels] + conv_channels
        pairs = list(zip(channels[:-1], channels[1:]))
        blocks = [
            TemporalConvNetBlock(in_c, out_c, kernel_size, padding, 2 ** i, dropout)
            for i, (in_c, out_c) in enumerate(pairs)
        ]
        super().__init__(*blocks, torch.nn.Conv1d(channels[-1], out_channels, 1))
