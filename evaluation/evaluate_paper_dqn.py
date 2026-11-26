
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.paper_dqn_agent import PaperDQNAgent
from enviroment.paper_sudoku_env import PaperSudokuEnv


DATA_PATH = ROOT / "dataset" / "test_curriculum.csv"
MODEL_PATH = ROOT / "models" / "sudoku_paper_dqn.pth"

df = pd.read_csv(DATA_PATH, dtype=str)

agent = PaperDQNAgent(
    state_size=81,
    action_size=729
)

agent.load(MODEL_PATH)
agent.epsilon = 0.0

results = []

for index, row in df.iterrows():

    puzzle = np.array(
        [int(x) for x in row["quizzes"]],
        dtype=np.int32
    ).reshape(9, 9)

    solution = np.array(
        [int(x) for x in row["solutions"]],
        dtype=np.int32
    ).reshape(9, 9)

    env = PaperSudokuEnv(
        puzzles=[puzzle],
        solutions=[solution],
        max_steps=100
    )

    state, info = env.reset()

    total_reward = 0
    done = False

    while not done:

        action = agent.select_action(
            state,
            info["occupied_mask"]
        )

        if action is None:
            break

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    solved = np.array_equal(env.board, solution)

    # Accuracy over originally empty cells only
    empty_cells = puzzle == 0
    total_empty = np.sum(empty_cells)

    correct_cells = np.sum(
        (env.board == solution) & empty_cells
    )

    accuracy = (
        correct_cells / total_empty * 100
        if total_empty > 0 else 100
    )

    level = row["level"] if "level" in row.index else "Unknown"

    results.append({
        "level": level,
        "solved": int(solved),
        "accuracy": accuracy,
        "reward": total_reward
    })

    if (index + 1) % 10 == 0:
        print(f"Evaluated: {index + 1}/{len(df)}")


# Overall results
results_df = pd.DataFrame(results)

print("\n========== PAPER DQN EVALUATION ==========")

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

print("\n========== RESULTS PER LEVEL ==========")

for level, group in results_df.groupby("level", sort=False):

    print(f"\nLevel {level}")

    print(
        "Solve Rate:",
        f"{group['solved'].mean() * 100:.2f}%"
    )

    print(
        "Cell Accuracy:",
        f"{group['accuracy'].mean():.2f}%"
    )


output_path = ROOT / "evaluation" / "paper_dqn_results.csv"

results_df.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
