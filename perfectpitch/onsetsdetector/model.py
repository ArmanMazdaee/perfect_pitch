import torch

from perfectpitch import constants


class OnsetsDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_pitches = constants.MAX_PITCH - constants.MIN_PITCH + 1
        self.conv2d = torch.nn.Sequential(
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=1, out_channels=32, kernel_size=3, padding=1
                )
            ),
            torch.nn.ReLU(),
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, padding=1
                )
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=3, padding=1
                )
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=(num_pitches // 4) * 64 + constants.POSENC_DIM,
                out_features=256,
            ),
            torch.nn.ReLU(),
        )
        self.sequential = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, dropout=0, activation="relu"
            ),
            num_layers=2,
            norm=torch.nn.LayerNorm(normalized_shape=256),
        )
        self.linear2 = torch.nn.Linear(in_features=256, out_features=num_pitches)

    def forward(self, spec, posenc, mask=None):
        if mask is not None:
            mask = ~mask.T

        conv2_input = spec.permute(1, 2, 0).unsqueeze(1)
        conv2_output = self.conv2d(conv2_input)
        linear1_input = torch.cat(
            [conv2_output.flatten(1, 2).permute(2, 0, 1), posenc], dim=2
        )
        linear1_output = self.linear1(linear1_input)
        sequential_output = self.sequential(linear1_output, src_key_padding_mask=mask)
        linear2_output = self.linear2(sequential_output)
        return linear2_output
