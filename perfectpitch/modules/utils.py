import torch


class Pad(torch.nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return torch.nn.functional.pad(x, self.padding)

    def extra_repr(self):
        return str(self.padding)
