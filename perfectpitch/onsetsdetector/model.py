import torch

from perfectpitch import constants


class _Conv2dResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation, dropout):
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
        self.dropout1 = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, dilation),
                dilation=(1, dilation),
            )
        )
        self.dropout2 = torch.nn.Dropout(dropout)

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
        x = self.dropout1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)

        res = self.downsample(input_) if self.downsample is not None else input_
        return torch.nn.functional.relu(res + x)


class OnsetsDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_pitches = constants.MAX_PITCH - constants.MIN_PITCH + 1
        self.conv2d_stack = torch.nn.Sequential(
            _Conv2dResidualBlock(
                in_channels=1, out_channels=8, dilation=1, dropout=0.2
            ),
            _Conv2dResidualBlock(
                in_channels=8, out_channels=16, dilation=2, dropout=0.2
            ),
            _Conv2dResidualBlock(
                in_channels=16, out_channels=32, dilation=4, dropout=0.2
            ),
            _Conv2dResidualBlock(
                in_channels=32, out_channels=64, dilation=8, dropout=0.2
            ),
            _Conv2dResidualBlock(
                in_channels=64, out_channels=64, dilation=16, dropout=0.2
            ),
        )
        self.linear_stack = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=num_pitches * 64, out_channels=128, kernel_size=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=128, out_channels=num_pitches, kernel_size=1),
        )

    def forward(self, spec):
        x = spec.unsqueeze(dim=1)
        x = self.conv2d_stack(x)
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.linear_stack(x)
        return x
