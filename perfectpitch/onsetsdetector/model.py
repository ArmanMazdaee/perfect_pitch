import torch
from axial_positional_embedding import AxialPositionalEmbedding
from reformer_pytorch import Reformer

from perfectpitch import constants


MAX_SEQ_LEN = 100000
DROPOUT = 0.1


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
        self.positional_embedding = AxialPositionalEmbedding(
            512, (MAX_SEQ_LEN // 64, 64)
        )
        self.sequential = Reformer(
            dim=512,
            depth=8,
            max_seq_len=MAX_SEQ_LEN,
            heads=8,
            lsh_dropout=DROPOUT,
            ff_dropout=DROPOUT,
        )
        self.linear = torch.nn.Linear(
            in_features=512, out_features=constants.MAX_PITCH - constants.MIN_PITCH + 1
        )

    def forward(self, spec, mask=None):
        conv1d_input = spec.transpose(1, 2)
        conv1d_output = self.conv1d(conv1d_input)
        sequential_input = self.positional_embedding(conv1d_output.transpose(1, 2))
        sequential_output = self.sequential(sequential_input, input_mask=mask)
        return self.linear(sequential_output)
