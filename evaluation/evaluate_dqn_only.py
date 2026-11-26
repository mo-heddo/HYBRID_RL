
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from enviroment.dqn_only_env import DQNOnlySudokuEnv
from algorithms.dqn_agent import DQNAgent


# ==============================
# Load the SAME 100 test puzzles
# ==============================

DATA_PATH = ROOT / "dataset" / "test_curriculum.csv"
MODEL_PATH = ROOT / "models" / "sudoku_dqn_only.pth"

df = pd.read_csv(DATA_PATH)

puzzles = np.array([
    np.array(list(map(int, p))).reshape(9, 9)
    for p in df["quizzes"]
])

solutions = np.array([
    np.array(list(map(int, s))).reshape(9, 9)
    for s in df["solutions"]
])


# ==============================
# Create environment
# ==============================

env = DQNOnlySudokuEnv(
    puzzles=puzzles,
    solutions=solutions,
    max_steps=100
)


# ==============================
# Load trained DQN-only model
# ==============================

agent = DQNAgent(
    state_size=81,
    action_size=729
)

agent.load(MODEL_PATH)
agent.epsilon = 0.0


# ==============================
# Evaluation
# ==============================

results = []

for episode in range(len(puzzles)):

    # Select exactly one test puzzle
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

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        episode_reward += reward

    final_board = env.board.copy()

    # Count only cells that were originally empty
    empty_positions = initial_board == 0
    total_empty = np.sum(empty_positions)

    correct_cells = np.sum(
        (final_board == solution) & empty_positions
    )

    accuracy = (
        correct_cells / total_empty * 100
        if total_empty > 0 else 100.0
    )

    solved = np.array_equal(final_board, solution)

    level = df.iloc[episode]["level"]

    results.append({
        "level": level,
        "solved": int(solved),
        "accuracy": accuracy,
        "correct_cells": correct_cells,
        "total_empty": total_empty,
        "reward": episode_reward
    })

    if (episode + 1) % 10 == 0:
        recent = results[-10:]

        print(
            f"Episode: {episode + 1}/{len(puzzles)} | "
            f"Solved: {sum(r['solved'] for r in results)} | "
            f"Recent Accuracy: "
            f"{np.mean([r['accuracy'] for r in recent]):.2f}%"
        )


# ==============================
# Overall results
# ==============================

results_df = pd.DataFrame(results)

print("\n========== DQN-ONLY EVALUATION ==========")

print("Total puzzles:", len(results_df))

print("Solved puzzles:", results_df["solved"].sum())

print(
    "Solve Rate:",
    f"{results_df['solved'].mean() * 100:.2f}%"
)

print(
    "Average Cell Accuracy:",
    f"{results_df['accuracy'].mean():.2f}%"
)

print(
    "Average Reward:",
    f"{results_df['reward'].mean():.2f}"
)


# ==============================
# Results per difficulty level
# ==============================

print("\n========== RESULTS PER LEVEL ==========")

for level, group in results_df.groupby("level", sort=False):

    print(f"\nLevel {level}")

    print(
        "Solved:",
        f"{group['solved'].sum()}/{len(group)}"
    )

    print(
        "Solve Rate:",
        f"{group['solved'].mean() * 100:.2f}%"
    )

    print(
        "Cell Accuracy:",
        f"{group['accuracy'].mean():.2f}%"
    )

    print(
        "Average Reward:",
        f"{group['reward'].mean():.2f}"
    )


# ==============================
# Save results
# ==============================

OUTPUT_PATH = ROOT / "evaluation" / "dqn_only_results.csv"

results_df.to_csv(OUTPUT_PATH, index=False)

print("\nResults saved to:", OUTPUT_PATH)
