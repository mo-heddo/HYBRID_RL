
import random
import numpy as np
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=10000):

        # Maximum number of experiences to store
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        next_action_mask
    ):

        experience = (
            state,
            action,
            reward,
            next_state,
            terminated,
            next_action_mask
        )

        self.buffer.append(experience)

    def sample(self, batch_size):

        # Randomly select experiences
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, terminated, next_masks = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(terminated, dtype=np.float32),
            np.array(next_masks, dtype=np.float32)
        )

    def __len__(self):

        return len(self.buffer)
