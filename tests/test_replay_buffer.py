import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from algorithms.replay_buffer import ReplayBuffer
import numpy as np

from algorithms.replay_buffer import ReplayBuffer


buffer = ReplayBuffer(capacity=100)

# Create example experiences
for i in range(20):

    state = np.zeros(83)

    action = i % 9

    reward = 1.0

    next_state = np.ones(83)

    terminated = False

    next_action_mask = np.array(
        [1, 0, 1, 0, 0, 0, 0, 0, 0]
    )

    buffer.add(
        state,
        action,
        reward,
        next_state,
        terminated,
        next_action_mask
    )


# Sample 5 experiences
batch = buffer.sample(5)

print("Buffer size:", len(buffer))

print("States shape:", batch[0].shape)
print("Actions shape:", batch[1].shape)
print("Rewards shape:", batch[2].shape)
print("Next states shape:", batch[3].shape)
print("Terminated shape:", batch[4].shape)
print("Next masks shape:", batch[5].shape)
