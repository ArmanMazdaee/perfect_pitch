import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dilation,
    ):
        super().__init__()
        self.conv1 = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, dilation),
                dilation=(1, dilation),
            )
        )
        self.dropout1 = torch.nn.Dropout(0.2)
        self.conv2 = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, dilation),
                dilation=(1, dilation),
            )
        )
        self.dropout2 = torch.nn.Dropout(0.2)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1
                )
            )

    def forward(self, spec):
        x = self.conv1(spec)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)

        res = self.downsample(spec) if self.downsample is not None else spec
        return torch.nn.functional.relu(res + x)


class _Conv2dStack(torch.nn.Sequential):
    def __init__(self, in_channels):
        super().__init__(
            _ResidualBlock(in_channels=in_channels, out_channels=16, dilation=1),
            _ResidualBlock(in_channels=16, out_channels=64, dilation=2),
            _ResidualBlock(in_channels=64, out_channels=64, dilation=4),
            _ResidualBlock(in_channels=64, out_channels=16, dilation=16),
            _ResidualBlock(in_channels=16, out_channels=1, dilation=32),
        )


class AcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.onsets_conv2d_stack = _Conv2dStack(in_channels=1)
        self.offsets_conv2d_stack = _Conv2dStack(in_channels=2)
        self.actives_conv2d_stack = _Conv2dStack(in_channels=2)

    def forward(self, spec):
        onsets_input = spec.unsqueeze(dim=1)
        onsets_output = self.onsets_conv2d_stack(onsets_input)

        onsets = torch.sigmoid(onsets_output).detach()
        offsets_input = torch.cat([onsets_input, onsets], dim=1)
        offsets_output = self.offsets_conv2d_stack(offsets_input)
        actives_output = self.actives_conv2d_stack(offsets_input)

        return {
            "onsets": onsets_output.squeeze(1),
            "offsets": offsets_output.squeeze(1),
            "actives": actives_output.squeeze(1),
        }
