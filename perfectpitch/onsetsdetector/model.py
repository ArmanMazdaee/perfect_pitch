import torch

from perfectpitch import constants


class _Conv2dResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
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
        self.conv2 = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
            )
        )

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1
                )
            )

    def forward(self, input_):
        x = self.conv1(input_)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)

        res = self.downsample(input_) if self.downsample is not None else input_
        return torch.nn.functional.relu(res + x)


class OnsetsDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_pitches = constants.MAX_PITCH - constants.MIN_PITCH + 1
        self.conv2d_stack = torch.nn.Sequential(
            _Conv2dResidualBlock(in_channels=1, out_channels=16, dilation=1),
            _Conv2dResidualBlock(in_channels=16, out_channels=16, dilation=2),
            _Conv2dResidualBlock(in_channels=16, out_channels=16, dilation=4),
            _Conv2dResidualBlock(in_channels=16, out_channels=16, dilation=8),
            _Conv2dResidualBlock(in_channels=16, out_channels=16, dilation=16),
            _Conv2dResidualBlock(in_channels=16, out_channels=16, dilation=32),
            _Conv2dResidualBlock(in_channels=16, out_channels=16, dilation=64),
            _Conv2dResidualBlock(in_channels=16, out_channels=16, dilation=128),
        )
        self.linear_stack = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=num_pitches * 16, out_channels=256, kernel_size=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=256, out_channels=num_pitches, kernel_size=1),
        )

    def forward(self, spec):
        x = spec.unsqueeze(dim=1)
        x = self.conv2d_stack(x)
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.linear_stack(x)
        return x
