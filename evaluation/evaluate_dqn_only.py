
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from enviroment.dqn_only_env import DQNOnlySudokuEnv
from algorithms.dqn_agent import DQNAgent


# Load 1000 unseen puzzles
df = pd.read_csv(
    "dataset/sudoku.csv",
    skiprows=range(1, 101),
    nrows=1000
)

puzzles = np.array([
    np.array(list(map(int, p))).reshape(9, 9)
    for p in df["quizzes"]
])

solutions = np.array([
    np.array(list(map(int, s))).reshape(9, 9)
    for s in df["solutions"]
])


# Create DQN-only environment
env = DQNOnlySudokuEnv(
    puzzles=puzzles,
    solutions=solutions,
    max_steps=100
)


# Load DQN-only model
agent = DQNAgent(
    state_size=81,
    action_size=729
)

agent.load("models/sudoku_dqn_only.pth")
agent.epsilon = 0.0


# Evaluation
solved_count = 0
total_cell_accuracy = 0
total_reward = 0

episodes = len(puzzles)

for episode in range(episodes):

    # Test each puzzle exactly once
    env.puzzles = puzzles[episode:episode + 1]
    env.solutions = solutions[episode:episode + 1]

    observation, info = env.reset()

    initial_board = observation.reshape(9, 9).copy()
    solution = solutions[episode]

    episode_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = agent.select_action(
            observation,
            info["action_mask"]
        )

        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward

    final_board = env.board

    empty_positions = initial_board == 0
    total_empty = np.sum(empty_positions)

    correct_cells = np.sum(
        (final_board == solution) & empty_positions
    )

    cell_accuracy = correct_cells / total_empty

    solved = np.array_equal(final_board, solution)

    if solved:
        solved_count += 1

    total_cell_accuracy += cell_accuracy
    total_reward += episode_reward

    if (episode + 1) % 100 == 0:
        print(
            f"Episode: {episode + 1}/{episodes} | "
            f"Solved: {solved_count} | "
            f"Cell Accuracy: "
            f"{total_cell_accuracy / (episode + 1) * 100:.2f}%"
        )


print("\n====== DQN ONLY EVALUATION ======")
print(f"Total puzzles: {episodes}")
print(f"Solved puzzles: {solved_count}")
print(f"Solve Rate: {solved_count / episodes * 100:.2f}%")
print(
    f"Average Cell Accuracy: "
    f"{total_cell_accuracy / episodes * 100:.2f}%"
)
print(f"Average Reward: {total_reward / episodes:.2f}")
print("================================")
