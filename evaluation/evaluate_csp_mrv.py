
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from enviroment.sudoku_env import SudokuEnv


# 1. Load 1000 unseen puzzles
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


# 2. Create environment
env = SudokuEnv(
    puzzles=puzzles,
    solutions=solutions,
    max_steps=100
)


# 3. Evaluation
episodes = len(puzzles)

solved_count = 0
total_cell_accuracy = 0
total_reward = 0

for episode in range(episodes):

    # Test each puzzle exactly once
    env.puzzles = puzzles[episode:episode + 1]
    env.solutions = solutions[episode:episode + 1]

    observation, info = env.reset()

    initial_board = observation.copy()
    solution = solutions[episode]

    episode_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Select the first available candidate.
        # No DQN is used in this experiment.
        valid_actions = np.where(info["action_mask"] == 1)[0]

        if len(valid_actions) == 0:
            break

        action = int(valid_actions[0])

        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward

    # Calculate accuracy
    final_board = env.board

    empty_positions = (initial_board == 0)
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


# 4. Final results
solve_rate = solved_count / episodes * 100
average_accuracy = total_cell_accuracy / episodes * 100
average_reward = total_reward / episodes

print("\n====== CSP + MRV (WITHOUT DQN) ======")
print(f"Total puzzles: {episodes}")
print(f"Solved puzzles: {solved_count}")
print(f"Solve Rate: {solve_rate:.2f}%")
print(f"Average Cell Accuracy: {average_accuracy:.2f}%")
print(f"Average Reward: {average_reward:.2f}")
print("====================================")
