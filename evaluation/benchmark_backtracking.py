
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.csp import SudokuCSP
from algorithms.dqn_agent import DQNAgent
from strategies.mrv import MRVStrategy


# -----------------------------------------
# Configuration
# -----------------------------------------

DATA_PATH = ROOT / "dataset" / "test_curriculum.csv"
MODEL_PATH = ROOT / "models" / "sudoku_dqn.pth"
OUTPUT_PATH = ROOT / "evaluation" / "backtracking_benchmark.csv"

REPEATS = 10

df = pd.read_csv(DATA_PATH)
mrv = MRVStrategy()

agent = DQNAgent(state_size=83, action_size=9)
agent.load(str(MODEL_PATH))
agent.q_network.eval()

# -----------------------------------------
# DQN state representation
# -----------------------------------------

def get_state(board, row, col):

    return np.concatenate([
        board.flatten() / 9.0,
        [row / 8.0, col / 8.0]
    ]).astype(np.float32)


# -----------------------------------------
# DQN candidate ranking
# -----------------------------------------

def rank_candidates(board, row, col, candidates):

    state = get_state(board, row, col)

    tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=agent.device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = agent.q_network(tensor)
        q_values = q_values.squeeze(0).cpu().numpy()

    return sorted(
        candidates,
        key=lambda digit: q_values[digit - 1],
        reverse=True
    )


# -----------------------------------------
# Common backtracking solver
# -----------------------------------------

def solve_sudoku(board, stats, use_dqn=False):

    stats["nodes"] += 1

    csp = SudokuCSP(board)
    all_candidates = csp.get_all_candidates()

    # Contradiction detected
    if any(len(values) == 0 for values in all_candidates.values()):
        return False

    # MRV cell selection
    cell, candidates = mrv.select_cell(board)

    if cell is None:
        return True

    row, col = cell

    # Candidate ordering
    if use_dqn:
        candidates = rank_candidates(
            board, row, col, candidates
        )

    # Otherwise, CSP returns candidates in ascending order

    for digit in candidates:

        board[row, col] = digit

        if solve_sudoku(board, stats, use_dqn):
            return True

        board[row, col] = 0
        stats["backtracks"] += 1

    return False


# -----------------------------------------
# Benchmark one solver on one puzzle
# -----------------------------------------

def benchmark_solver(puzzle, solution, use_dqn):

    times = []
    node_counts = []
    backtrack_counts = []
    correct = True

    for _ in range(REPEATS):

        board = puzzle.copy()

        stats = {
            "nodes": 0,
            "backtracks": 0
        }

        start = time.perf_counter()

        solved = solve_sudoku(board, stats, use_dqn)

        elapsed = time.perf_counter() - start

        times.append(elapsed)
        node_counts.append(stats["nodes"])
        backtrack_counts.append(stats["backtracks"])

        if not solved or not np.array_equal(board, solution):
            correct = False

    return {
        "solved": correct,
        "avg_time": np.mean(times),
        "median_time": np.median(times),
        "avg_nodes": np.mean(node_counts),
        "avg_backtracks": np.mean(backtrack_counts)
    }


# -----------------------------------------
# Warm up DQN before timing
# -----------------------------------------

dummy_board = np.zeros((9, 9), dtype=np.int32)
rank_candidates(dummy_board, 0, 0, [1, 2, 3])

if torch.cuda.is_available():
    torch.cuda.synchronize()


# -----------------------------------------
# Run benchmark
# -----------------------------------------

results = []

print("Starting benchmark...")
print(f"Test puzzles: {len(df)}")
print(f"Repetitions per puzzle: {REPEATS}")

for index, row in df.iterrows():

    puzzle = np.array(
        [int(x) for x in str(row["quizzes"])],
        dtype=np.int32
    ).reshape(9, 9)

    solution = np.array(
        [int(x) for x in str(row["solutions"])],
        dtype=np.int32
    ).reshape(9, 9)

    empty_count = int(np.sum(puzzle == 0))

    level = {
        5: 1,
        15: 2,
        25: 3,
        35: 4,
        45: 5
    }.get(empty_count, 0)

    # Traditional CSP + MRV + Backtracking
    traditional = benchmark_solver(
        puzzle, solution, use_dqn=False
    )

    # DQN-guided CSP + MRV + Backtracking
    hybrid = benchmark_solver(
        puzzle, solution, use_dqn=True
    )

    results.append({
        "Puzzle": index + 1,
        "Level": level,

        "Traditional Solved": int(traditional["solved"]),
        "Traditional Time": traditional["avg_time"],
        "Traditional Median Time": traditional["median_time"],
        "Traditional Nodes": traditional["avg_nodes"],
        "Traditional Backtracks": traditional["avg_backtracks"],

        "Hybrid Solved": int(hybrid["solved"]),
        "Hybrid Time": hybrid["avg_time"],
        "Hybrid Median Time": hybrid["median_time"],
        "Hybrid Nodes": hybrid["avg_nodes"],
        "Hybrid Backtracks": hybrid["avg_backtracks"]
    })

    if (index + 1) % 10 == 0:
        print(f"Completed {index + 1}/{len(df)} puzzles")


# -----------------------------------------
# Save detailed results
# -----------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)


# -----------------------------------------
# Overall comparison
# -----------------------------------------

print("\n========== OVERALL BENCHMARK ==========")

for name, prefix in [
    ("CSP + MRV + Backtracking", "Traditional"),
    ("DQN-Guided Hybrid Backtracking", "Hybrid")
]:

    solved = results_df[f"{prefix} Solved"].sum()

    print(f"\n{name}")
    print(f"Solve Rate: {solved / len(df) * 100:.2f}%")
    print(
        f"Average Time: "
        f"{results_df[f'{prefix} Time'].mean():.6f} seconds"
    )
    print(
        f"Median Time: "
        f"{results_df[f'{prefix} Median Time'].median():.6f} seconds"
    )
    print(
        f"Average Nodes: "
        f"{results_df[f'{prefix} Nodes'].mean():.2f}"
    )
    print(
        f"Average Backtracks: "
        f"{results_df[f'{prefix} Backtracks'].mean():.2f}"
    )

print("\n========== RESULTS PER LEVEL ==========")

for level in sorted(results_df["Level"].unique()):

    if level == 0:
        continue

    subset = results_df[results_df["Level"] == level]

    print(f"\nLevel {level}")

    print(
        "Traditional time:",
        f"{subset['Traditional Time'].mean():.6f} seconds"
    )

    print(
        "Hybrid time:",
        f"{subset['Hybrid Time'].mean():.6f} seconds"
    )

    print(
        "Traditional nodes:",
        f"{subset['Traditional Nodes'].mean():.2f}"
    )

    print(
        "Hybrid nodes:",
        f"{subset['Hybrid Nodes'].mean():.2f}"
    )

print("\nSaved results to:")
print(OUTPUT_PATH)
