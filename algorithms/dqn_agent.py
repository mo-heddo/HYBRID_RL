
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.dqn import QNetwork
from algorithms.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        state_size=83,
        action_size=9,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_capacity=10000,
        target_update=100
    ):

        # Device: GPU if available, otherwise CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.action_size = action_size

        # Discount factor
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Training parameters
        self.batch_size = batch_size
        self.target_update = target_update

        self.learn_steps = 0

        # Main Q-Network
        self.q_network = QNetwork(
            state_size, action_size
        ).to(self.device)

        # Target Network
        self.target_network = QNetwork(
            state_size, action_size
        ).to(self.device)

        # Initially copy main network weights
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        # Target network is only used for prediction
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        # Loss function
        self.loss_function = nn.MSELoss()

        # Experience Replay
        self.memory = ReplayBuffer(buffer_capacity)

    # ---------------------------------
    # Select an action using action mask
    # ---------------------------------

    def select_action(self, state, action_mask):

        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            raise ValueError("No valid actions available.")

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:

            # Random valid action
            return int(random.choice(valid_actions))

        # Exploitation: choose the best Q-value
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.q_network(state_tensor)

            # Block invalid actions
            mask_tensor = torch.tensor(
                action_mask,
                dtype=torch.bool
            ).to(self.device)

            q_values[0][~mask_tensor] = -float("inf")

            action = torch.argmax(q_values, dim=1).item()

        return int(action)

    # ---------------------------------
    # Store experience
    # ---------------------------------

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        next_action_mask
    ):

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            terminated,
            next_action_mask
        )

    # ---------------------------------
    # Train the Q-Network
    # ---------------------------------

    def train_step(self):

        # Wait until enough experiences exist
        if len(self.memory) < self.batch_size:
            return None

        # Sample random experiences
        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            next_masks
        ) = self.memory.sample(self.batch_size)

        # Convert to PyTorch tensors
        states = torch.tensor(
            states, dtype=torch.float32
        ).to(self.device)

        actions = torch.tensor(
            actions, dtype=torch.long
        ).unsqueeze(1).to(self.device)

        rewards = torch.tensor(
            rewards, dtype=torch.float32
        ).to(self.device)

        next_states = torch.tensor(
            next_states, dtype=torch.float32
        ).to(self.device)

        terminated = torch.tensor(
            terminated, dtype=torch.float32
        ).to(self.device)

        next_masks = torch.tensor(
            next_masks, dtype=torch.bool
        ).to(self.device)

        # Current Q-values for selected actions
        current_q = self.q_network(states).gather(
            1, actions
        ).squeeze(1)

        # Calculate target Q-values
        with torch.no_grad():

            next_q = self.target_network(next_states)

            # Mask invalid next actions
            next_q = next_q.masked_fill(
                ~next_masks, -float("inf")
            )

            # Handle terminal states or states with no valid actions
            has_valid_action = next_masks.any(dim=1)

            max_next_q = next_q.max(dim=1)[0]

            max_next_q = torch.where(
                has_valid_action,
                max_next_q,
                torch.zeros_like(max_next_q)
            )

            # Bellman equation
            target_q = rewards + (
                self.gamma * max_next_q * (1 - terminated)
            )

        # Calculate loss
        loss = self.loss_function(
            current_q,
            target_q
        )

        # Update network weights
        self.optimizer.zero_grad()

        loss.backward()

        # Prevent excessively large gradients
        nn.utils.clip_grad_norm_(
            self.q_network.parameters(), 10.0
        )

        self.optimizer.step()

        self.learn_steps += 1

        # Update target network periodically
        if self.learn_steps % self.target_update == 0:

            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )

        return loss.item()

    # ---------------------------------
    # Reduce exploration over time
    # ---------------------------------

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # ---------------------------------
    # Save trained model
    # ---------------------------------

    def save(self, path):

        torch.save(
            self.q_network.state_dict(),
            path
        )

    # ---------------------------------
    # Load trained model
    # ---------------------------------

    def load(self, path):

        self.q_network.load_state_dict(
            torch.load(path, map_location=self.device)
        )

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )
