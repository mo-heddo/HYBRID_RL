
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from enviroment.sudoku_env import SudokuEnv
from algorithms.dqn_agent import DQNAgent


# 1. Load 100 puzzles AFTER the first 100 training puzzles
df = pd.read_csv(
    "dataset/sudoku.csv",
    skiprows=range(1, 101),
    nrows=100
)

puzzles = np.array([
    np.array(list(map(int, p))).reshape(9, 9)
    for p in df["quizzes"]
])

solutions = np.array([
    np.array(list(map(int, s))).reshape(9, 9)
    for s in df["solutions"]
])


# 2. Create the evaluation environment
env = SudokuEnv(
    puzzles=puzzles,
    solutions=solutions,
    max_steps=100
)


# 3. Load the trained DQN
agent = DQNAgent(
    state_size=83,
    action_size=9
)

agent.load("models/sudoku_dqn.pth")

# Disable exploration
agent.epsilon = 0.0


# 4. State encoding (same as training)
def encode_state(board, current_cell):

    board_features = board.flatten() / 9.0

    if current_cell is None:
        row, col = 0, 0
    else:
        row, col = current_cell

    row = row / 8.0
    col = col / 8.0

    state = np.concatenate([
        board_features,
        [row, col]
    ])

    return state.astype(np.float32)


# 5. Evaluation
episodes = 100

solved_count = 0
total_cell_accuracy = 0
total_reward = 0

for episode in range(episodes):

    observation, info = env.reset()

    initial_board = observation.copy()
    solution = env.solution.copy()

    state = encode_state(
        observation,
        info["current_cell"]
    )

    action_mask = info["action_mask"]

    episode_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = agent.select_action(
            state,
            action_mask
        )

        next_observation, reward, terminated, truncated, next_info = env.step(action)

        next_state = encode_state(
            next_observation,
            next_info["current_cell"]
        )

        state = next_state
        action_mask = next_info["action_mask"]

        episode_reward += reward

    # 6. Calculate evaluation metrics
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

    if (episode + 1) % 10 == 0:
        print(
            f"Episode: {episode + 1}/{episodes} | "
            f"Solved: {solved_count} | "
            f"Cell Accuracy: "
            f"{total_cell_accuracy / (episode + 1) * 100:.2f}%"
        )


# 7. Final results
solve_rate = (solved_count / episodes) * 100
average_cell_accuracy = (total_cell_accuracy / episodes) * 100
average_reward = total_reward / episodes

print("\n========== EVALUATION RESULTS ==========")
print(f"Total puzzles: {episodes}")
print(f"Solved puzzles: {solved_count}")
print(f"Solve Rate: {solve_rate:.2f}%")
print(f"Average Cell Accuracy: {average_cell_accuracy:.2f}%")
print(f"Average Reward: {average_reward:.2f}")
print("========================================")
