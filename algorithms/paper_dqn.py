
import torch
import torch.nn as nn


class PaperQNetwork(nn.Module):
    def __init__(self, action_size=729):
        super().__init__()

        # Input: 1 x 9 x 9 Sudoku board
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        # 64 feature maps, each 9 x 9
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, state):
        # Support a single board or a batch of boards
        if state.dim() == 2:
            state = state.unsqueeze(0)

        if state.dim() == 3:
            state = state.unsqueeze(1)

        state = state.float()

        features = self.conv_layers(state)
        q_values = self.fc_layers(features)

        return q_values
