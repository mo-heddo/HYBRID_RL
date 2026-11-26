
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from enviroment.sudoku_env import SudokuEnv
from algorithms.dqn_agent import DQNAgent


# ---------------------------------------
# 1. Load Unseen Curriculum Test Dataset
# ---------------------------------------

df = pd.read_csv("dataset/test_curriculum.csv")

puzzles = np.array([
    np.array(list(map(int, p))).reshape(9, 9)
    for p in df["quizzes"]
])

solutions = np.array([
    np.array(list(map(int, s))).reshape(9, 9)
    for s in df["solutions"]
])

print("Total test puzzles:", len(puzzles))


# ---------------------------------------
# 2. Create Evaluation Environment
# ---------------------------------------

env = SudokuEnv(
    puzzles=puzzles,
    solutions=solutions,
    max_steps=100
)


# ---------------------------------------
# 3. Load Original Hybrid DQN
# ---------------------------------------

agent = DQNAgent(
    state_size=83,
    action_size=9
)

agent.load("models/sudoku_dqn.pth")

# Disable exploration
agent.epsilon = 0.0


# ---------------------------------------
# 4. State Encoding (Same as Training)
# ---------------------------------------

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


# ---------------------------------------
# 5. Evaluation
# ---------------------------------------

results = []

total_solved = 0
total_accuracy = 0
total_reward = 0

for episode in range(len(df)):

    # Force evaluation to use this exact puzzle
    env.puzzles = puzzles[episode:episode + 1]
    env.solutions = solutions[episode:episode + 1]

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

        next_observation, reward, terminated, truncated, next_info = (
            env.step(action)
        )

        state = encode_state(
            next_observation,
            next_info["current_cell"]
        )

        action_mask = next_info["action_mask"]

        episode_reward += reward

    # -----------------------------------
    # 6. Calculate Metrics
    # -----------------------------------

    final_board = env.board

    empty_positions = initial_board == 0
    total_empty = np.sum(empty_positions)

    correct_cells = np.sum(
        (final_board == solution) & empty_positions
    )

    cell_accuracy = (
        correct_cells / total_empty
        if total_empty > 0 else 1.0
    )

    solved = np.array_equal(final_board, solution)

    if solved:
        total_solved += 1

    total_accuracy += cell_accuracy
    total_reward += episode_reward

    results.append({
        "level": df.iloc[episode]["level"],
        "solved": int(solved),
        "cell_accuracy": cell_accuracy * 100,
        "reward": episode_reward
    })

    if (episode + 1) % 10 == 0:

        print(
            f"Puzzle: {episode + 1}/{len(df)} | "
            f"Solved: {total_solved} | "
            f"Average Accuracy: "
            f"{total_accuracy / (episode + 1) * 100:.2f}%"
        )


# ---------------------------------------
# 7. Results By Difficulty Level
# ---------------------------------------

results_df = pd.DataFrame(results)

print("\n========== RESULTS BY LEVEL ==========")

for level, group in results_df.groupby("level", sort=True):

    count = len(group)
    solved = group["solved"].sum()

    solve_rate = (solved / count) * 100
    accuracy = group["cell_accuracy"].mean()
    avg_reward = group["reward"].mean()

    print(f"\n{level}")
    print(f"Total puzzles: {count}")
    print(f"Solved: {solved}/{count}")
    print(f"Solve Rate: {solve_rate:.2f}%")
    print(f"Average Cell Accuracy: {accuracy:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")


# ---------------------------------------
# 8. Overall Results
# ---------------------------------------

episodes = len(results_df)

solve_rate = (total_solved / episodes) * 100
average_accuracy = (total_accuracy / episodes) * 100
average_reward = total_reward / episodes

print("\n========== ORIGINAL HYBRID RESULTS ==========")

print(f"Total puzzles: {episodes}")
print(f"Solved puzzles: {total_solved}")
print(f"Solve Rate: {solve_rate:.2f}%")
print(f"Average Cell Accuracy: {average_accuracy:.2f}%")
print(f"Average Reward: {average_reward:.2f}")

print("=============================================")


# ---------------------------------------
# 9. Save Results
# ---------------------------------------

results_df.to_csv(
    "evaluation/original_hybrid_test_results.csv",
    index=False
)

print("\nResults saved to:")
print("evaluation/original_hybrid_test_results.csv")
