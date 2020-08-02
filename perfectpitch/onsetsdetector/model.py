import math

import torch

from perfectpitch import constants


DROPOUT = 0.1


def positional_encoding(x):
    length = x.shape[0]
    dimension = x.shape[2]
    position = torch.arange(0, length, 1, dtype=torch.float, device=x.device)
    div_term = torch.arange(0, dimension, 2, dtype=torch.float, device=x.device)
    points = position.unsqueeze(1) * torch.exp(
        div_term * -math.log(10000.0) / dimension
    )
    sin = torch.sin(points)
    cos = torch.cos(points)
    encoding = torch.cat([sin, cos], dim=1).unsqueeze(1)
    return x + encoding


class TransformerConvEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_conv, dropout):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)

        self.conv1ds = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=d_model, out_channels=dim_conv, kernel_size=3, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(
                in_channels=dim_conv, out_channels=d_model, kernel_size=3, padding=1
            ),
        )
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = self.self_attn(
            query=src,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        x = self.dropout1(x)
        x = self.norm1(src + x)

        y = x.permute(1, 2, 0)
        y = self.conv1ds(y)
        y = y.permute(2, 0, 1)
        y = self.dropout2(y)
        return self.norm2(x + y)


class OnsetsDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=constants.SPEC_DIM,
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
        )
        self.sequential = torch.nn.TransformerEncoder(
            encoder_layer=TransformerConvEncoderLayer(
                d_model=512, nhead=4, dim_conv=2048, dropout=DROPOUT,
            ),
            num_layers=8,
        )
        self.linear = torch.nn.Linear(
            in_features=512, out_features=constants.MAX_PITCH - constants.MIN_PITCH + 1
        )

    def forward(self, spec, mask=None):
        if mask is not None:
            mask = ~mask.T

        conv1d_input = spec.permute(1, 2, 0)
        conv1d_output = self.conv1d(conv1d_input)
        sequential_input = positional_encoding(conv1d_output.permute(2, 0, 1))
        sequential_output = self.sequential(sequential_input, src_key_padding_mask=mask)
        return self.linear(sequential_output)
