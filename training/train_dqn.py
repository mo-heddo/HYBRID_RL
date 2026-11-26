
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from enviroment.sudoku_env import SudokuEnv
from algorithms.dqn_agent import DQNAgent


# ---------------------------------------
# 1. Load Sudoku Dataset
# ---------------------------------------

df = pd.read_csv("dataset/sudoku.csv").head(100)

puzzles = np.array([
    np.array(list(map(int, p))).reshape(9, 9)
    for p in df["quizzes"]
])

solutions = np.array([
    np.array(list(map(int, s))).reshape(9, 9)
    for s in df["solutions"]
])


# ---------------------------------------
# 2. Create Environment
# ---------------------------------------

env = SudokuEnv(
    puzzles=puzzles,
    solutions=solutions,
    max_steps=100
)


# ---------------------------------------
# 3. Create DQN Agent
# ---------------------------------------

agent = DQNAgent(
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
)


# ---------------------------------------
# 4. Convert Sudoku Board to State
# ---------------------------------------

def encode_state(board, current_cell):

    # Flatten 9x9 board into 81 values
    board_features = board.flatten() / 9.0

    # Encode selected cell coordinates
    if current_cell is None:
        row, col = 0, 0
    else:
        row, col = current_cell

    row = row / 8.0
    col = col / 8.0

    # Combine all features
    state = np.concatenate([
        board_features,
        [row, col]
    ])

    return state.astype(np.float32)


# ---------------------------------------
# 5. Training Configuration
# ---------------------------------------

episodes = 1000

episode_rewards = []
episode_losses = []


# ---------------------------------------
# 6. Training Loop
# ---------------------------------------

for episode in range(episodes):

    observation, info = env.reset()

    state = encode_state(
        observation,
        info["current_cell"]
    )

    action_mask = info["action_mask"]

    total_reward = 0
    losses = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Select action using DQN and action masking
        action = agent.select_action(
            state,
            action_mask
        )

        # Execute action in Sudoku environment
        next_observation, reward, terminated, truncated, next_info = (
            env.step(action)
        )

        # Encode next state
        next_state = encode_state(
            next_observation,
            next_info["current_cell"]
        )

        next_action_mask = next_info["action_mask"]

        # Store experience
        agent.remember(
            state,
            action,
            reward,
            next_state,
            terminated or truncated,
            next_action_mask
        )

        # Train DQN
        loss = agent.train_step()

        if loss is not None:
            losses.append(loss)

        # Move to next state
        state = next_state
        action_mask = next_action_mask

        total_reward += reward

    # Reduce exploration after each episode
    agent.decay_epsilon()

    # Record performance
    episode_rewards.append(total_reward)

    average_loss = (
        np.mean(losses) if losses else 0.0
    )

    episode_losses.append(average_loss)

    # Print progress
    if (episode + 1) % 10 == 0:

        avg_reward = np.mean(episode_rewards[-10:])

        print(
            f"Episode: {episode + 1}/{episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Average Reward: {avg_reward:.2f} | "
            f"Loss: {average_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )


# ---------------------------------------
# 7. Save Trained Model
# ---------------------------------------

Path("models").mkdir(exist_ok=True)

agent.save("models/sudoku_dqn.pth")

print("\nTraining completed!")
print("Model saved to models/sudoku_dqn.pth")
