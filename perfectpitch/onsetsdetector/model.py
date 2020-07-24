import torch

from perfectpitch import constants


DROPOUT = 0.0


class _Conv2dResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.dropout1 = torch.nn.Dropout(DROPOUT)

        self.conv2 = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.dropout2 = torch.nn.Dropout(DROPOUT)

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
        self.conv2d = torch.nn.Sequential(
            _Conv2dResidualBlock(in_channels=1, out_channels=48),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),
            _Conv2dResidualBlock(in_channels=48, out_channels=96),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=(constants.SPEC_DIM // 4) * 96,
                out_features=512 - constants.POSENC_DIM,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=DROPOUT),
        )
        self.sequential = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=512,
                nhead=4,
                dim_feedforward=1024,
                dropout=DROPOUT,
                activation="relu",
            ),
            num_layers=4,
        )
        self.linear2 = torch.nn.Linear(in_features=512, out_features=num_pitches)

    def forward(self, spec, posenc, mask=None):
        if mask is not None:
            mask = ~mask.T

        conv2_input = spec.permute(1, 2, 0).unsqueeze(1)
        conv2_output = self.conv2d(conv2_input)
        linear1_input = conv2_output.flatten(1, 2).permute(2, 0, 1)
        linear1_output = self.linear1(linear1_input)
        sequential_input = torch.cat([linear1_output, posenc], dim=2)
        sequential_output = self.sequential(sequential_input, src_key_padding_mask=mask)
        linear2_output = self.linear2(sequential_output)
        return linear2_output
