import torch
import torch.nn as nn

from perfectpitch import constants


class _Conv2dStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, padding=1)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=48, out_channels=48, kernel_size=3, padding=1
        )
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(
            in_channels=48, out_channels=96, kernel_size=3, padding=1
        )
        self.activation3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout3 = nn.Dropout(0.25)

        self.linear = nn.Conv1d(
            in_channels=96 * (constants.SPEC_N_BINS // 4),
            out_channels=768,
            kernel_size=1,
        )
        self.activation_linear = nn.ReLU()
        self.dropout_linear = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.activation3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)

        x = self.linear(x)
        x = self.activation_linear(x)
        x = self.dropout_linear(x)
        return x


class AcousticModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.onsets_conv2d_stack = _Conv2dStack()
        self.onsets_linear = torch.nn.Conv1d(
            in_channels=768, out_channels=88, kernel_size=1
        )

        self.offsets_conv2d_stack = _Conv2dStack()
        self.offsets_linear = torch.nn.Conv1d(
            in_channels=768, out_channels=88, kernel_size=1
        )

        self.actives_conv2d_stack = _Conv2dStack()
        self.actives_linear = torch.nn.Conv1d(
            in_channels=768, out_channels=88, kernel_size=1
        )

    def forward(self, spec):
        onsets = self.onsets_conv2d_stack(spec)
        onsets = self.onsets_linear(onsets)

        offsets = self.offsets_conv2d_stack(spec)
        offsets = self.offsets_linear(offsets)

        actives = self.actives_conv2d_stack(spec)
        actives = self.actives_linear(actives)

        return {"onsets": onsets, "offsets": offsets, "actives": actives}
