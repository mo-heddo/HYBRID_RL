
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from algorithms.paper_dqn import PaperQNetwork


class PaperDQNAgent:

    def __init__(
        self,
        state_size=81,
        action_size=729,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=256,
        buffer_size=100000,
        target_update=1000
    ):

        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_network = PaperQNetwork(action_size).to(self.device)

        self.target_network = PaperQNetwork(action_size).to(self.device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()

        self.memory = deque(maxlen=buffer_size)

        self.training_steps = 0

    def select_action(self, state, occupied_mask):

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).reshape(
            1, 9, 9
        ).to(self.device)

        # Get available actions
        available_actions = np.where(occupied_mask == 0)[0]

        if len(available_actions) == 0:
            return None

        # Exploration
        if random.random() < self.epsilon:
            return int(random.choice(available_actions))

        # Exploitation
        with torch.no_grad():

            q_values = self.q_network(state_tensor).cpu().numpy()[0]

        # Try actions from highest Q-value to lowest
        sorted_actions = np.argsort(q_values)[::-1]

        for action in sorted_actions:

            if occupied_mask[action] == 0:
                return int(action)

        return None

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_occupied_mask
    ):

        self.memory.append((
            np.array(state, dtype=np.float32).reshape(9, 9),
            action,
            reward,
            np.array(next_state, dtype=np.float32).reshape(9, 9),
            done,
            np.array(next_occupied_mask, dtype=np.int8)
        ))

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        states = torch.FloatTensor(
            np.array(states)
        ).unsqueeze(1).to(self.device)

        next_states = torch.FloatTensor(
            np.array(next_states)
        ).unsqueeze(1).to(self.device)

        actions = torch.LongTensor(actions).to(self.device)

        rewards = torch.FloatTensor(rewards).to(self.device)

        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():

            next_q_values = self.target_network(next_states)

            next_masks = torch.BoolTensor(
                np.array(next_masks)
            ).to(self.device)

            next_q_values[next_masks.bool()] = -torch.inf

            max_next_q = next_q_values.max(dim=1)[0]

            max_next_q[torch.isinf(max_next_q)] = 0

            target_q = rewards + (
                self.gamma * max_next_q * (1 - dones)
            )

        loss = self.loss_function(current_q, target_q)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update == 0:

            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )

        return loss.item()

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def save(self, path):

        torch.save(
            self.q_network.state_dict(),
            path
        )

    def load(self, path):

        state_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        self.q_network.load_state_dict(state_dict)

        self.target_network.load_state_dict(state_dict)
