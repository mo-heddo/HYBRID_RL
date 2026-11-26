
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import torch
from algorithms.paper_dqn import PaperQNetwork

model = PaperQNetwork()

# Create a sample Sudoku board
board = torch.zeros((9, 9))

# Test one board
output = model(board)

print("Single board output:", output.shape)

# Test a batch of 32 boards
batch = torch.zeros((32, 9, 9))
batch_output = model(batch)

print("Batch output:", batch_output.shape)

print("Total parameters:", sum(
    p.numel() for p in model.parameters()
))
