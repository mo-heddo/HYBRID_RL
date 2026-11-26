
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from enviroment.dqn_only_env import DQNOnlySudokuEnv
from algorithms.dqn_agent import DQNAgent


# Load the same 100 training puzzles
df = pd.read_csv("dataset/sudoku.csv").head(100)

puzzles = np.array([
    np.array(list(map(int, p))).reshape(9, 9)
    for p in df["quizzes"]
])

solutions = np.array([
    np.array(list(map(int, s))).reshape(9, 9)
    for s in df["solutions"]
])


env = DQNOnlySudokuEnv(
    puzzles=puzzles,
    solutions=solutions,
    max_steps=100
)


# DQN with 81 input features and 729 actions
agent = DQNAgent(
    state_size=81,
    action_size=729,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    batch_size=64,
    buffer_capacity=10000,
    target_update=100
)


episodes = 1000

episode_rewards = []
episode_losses = []

for episode in range(episodes):

    state, info = env.reset()

    action_mask = info["action_mask"]

    total_reward = 0
    losses = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = agent.select_action(state, action_mask)

        next_state, reward, terminated, truncated, next_info = env.step(action)

        next_action_mask = next_info["action_mask"]

        agent.remember(
            state,
            action,
            reward,
            next_state,
            terminated or truncated,
            next_action_mask
        )

        loss = agent.train_step()

        if loss is not None:
            losses.append(loss)

        state = next_state
        action_mask = next_action_mask

        total_reward += reward

    agent.decay_epsilon()

    episode_rewards.append(total_reward)

    average_loss = np.mean(losses) if losses else 0.0
    episode_losses.append(average_loss)

    if (episode + 1) % 10 == 0:

        avg_reward = np.mean(episode_rewards[-10:])

        print(
            f"Episode: {episode + 1}/{episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Average Reward: {avg_reward:.2f} | "
            f"Loss: {average_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )


Path("models").mkdir(exist_ok=True)

agent.save("models/sudoku_dqn_only.pth")

print("\nDQN-only training completed!")
print("Model saved to models/sudoku_dqn_only.pth")
