import torch

from perfectpitch import constants


DROPOUT = 0.0


class TransformerConvEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_conv, dropout):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)

        self.conv1ds = torch.nn.Sequential(
            torch.nn.Conv1d(d_model, dim_conv, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(dim_conv, d_model, kernel_size=3, padding=1),
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
        num_pitches = constants.MAX_PITCH - constants.MIN_PITCH + 1
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),
            torch.nn.Dropout(p=DROPOUT),
            torch.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),
            torch.nn.Dropout(p=DROPOUT),
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
            encoder_layer=TransformerConvEncoderLayer(
                d_model=512, nhead=4, dim_conv=512, dropout=DROPOUT,
            ),
            num_layers=6,
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
