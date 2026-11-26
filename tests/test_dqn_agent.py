
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from algorithms.dqn_agent import DQNAgent


agent = DQNAgent(batch_size=4)

# Example Sudoku state
state = np.zeros(83, dtype=np.float32)

# Only digits 2, 5, and 8 are valid
action_mask = np.array(
    [0, 1, 0, 0, 1, 0, 0, 1, 0]
)

# Select an action
action = agent.select_action(state, action_mask)

print("Selected action:", action)
print("Selected digit:", action + 1)

# Store some example experiences
for i in range(10):

    next_state = np.ones(83, dtype=np.float32)

    next_mask = np.array(
        [1, 0, 1, 0, 0, 0, 0, 0, 0]
    )

    agent.remember(
        state,
        action,
        1.0,
        next_state,
        False,
        next_mask
    )

# Train the agent
loss = agent.train_step()

print("Training loss:", loss)

# Reduce exploration
agent.decay_epsilon()

print("Epsilon:", agent.epsilon)
