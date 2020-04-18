import torch

from perfectpitch import constants


class _Conv2dStack(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=48, kernel_size=3, padding=1
        )
        self.activation1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(
            in_channels=48, out_channels=48, kernel_size=3, padding=1
        )
        self.activation2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout2 = torch.nn.Dropout(0.25)

        self.conv3 = torch.nn.Conv2d(
            in_channels=48, out_channels=96, kernel_size=3, padding=1
        )
        self.activation3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout3 = torch.nn.Dropout(0.25)

        self.linear = torch.nn.Conv1d(
            in_channels=96 * (constants.SPEC_N_BINS // 4),
            out_channels=768,
            kernel_size=1,
        )
        self.activation_linear = torch.nn.ReLU()
        self.dropout_linear = torch.nn.Dropout(0.5)

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
