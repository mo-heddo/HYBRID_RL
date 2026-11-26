
import torch

from algorithms.dqn import QNetwork

# Create the network
model = QNetwork()

# Example state: 81 board values + row + column
state = torch.zeros(83, dtype=torch.float32)

# Get Q-values
q_values = model(state)

print("Q-values:", q_values)
print("Output shape:", q_values.shape)
