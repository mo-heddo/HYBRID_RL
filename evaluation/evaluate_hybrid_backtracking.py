
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.csp import SudokuCSP
from algorithms.dqn_agent import DQNAgent
from strategies.mrv import MRVStrategy


# -----------------------------------------
# 1. Configuration
# -----------------------------------------

DATA_PATH = ROOT / "dataset" / "test_curriculum.csv"
MODEL_PATH = ROOT / "models" / "sudoku_dqn.pth"

OUTPUT_PATH = (
    ROOT / "evaluation" / "hybrid_backtracking_results.csv"
)

df = pd.read_csv(DATA_PATH)

mrv = MRVStrategy()

# Load the existing original hybrid DQN
agent = DQNAgent(state_size=83, action_size=9)
agent.load(str(MODEL_PATH))

agent.q_network.eval()

# Make sure the model is used greedily (no exploration)
if hasattr(agent, "epsilon"):
    agent.epsilon = 0.0


# -----------------------------------------
# 2. DQN state representation
# -----------------------------------------

def get_state(board, row, col):

    normalized_board = board.flatten() / 9.0

    state = np.concatenate([
        normalized_board,
        [row / 8.0, col / 8.0]
    ])

    return state.astype(np.float32)


# -----------------------------------------
# 3. Rank candidates using DQN
# -----------------------------------------

def rank_candidates(board, row, col, candidates):

    state = get_state(board, row, col)

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=agent.device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = agent.q_network(state_tensor)
        q_values = q_values.squeeze(0).cpu().numpy()

    # Highest Q-value first, considering only legal candidates
    ranked_candidates = sorted(
        candidates,
        key=lambda digit: q_values[digit - 1],
        reverse=True
    )

    return ranked_candidates


# -----------------------------------------
# 4. DQN-Guided Backtracking Solver
# -----------------------------------------

def solve_sudoku(board, stats):

    stats["nodes"] += 1

    csp = SudokuCSP(board)

    # Stop if there is a contradiction
    all_candidates = csp.get_all_candidates()

    if any(len(values) == 0 for values in all_candidates.values()):
        return False

    # MRV selects the cell with the fewest candidates
    cell, candidates = mrv.select_cell(board)

    # No empty cells: solved
    if cell is None:
        return True

    row, col = cell

    # DQN ranks the legal candidates
    ranked_candidates = rank_candidates(
        board, row, col, candidates
    )

    # Try candidates in DQN's preferred order
    for digit in ranked_candidates:

        board[row, col] = digit

        if solve_sudoku(board, stats):
            return True

        # Undo the unsuccessful choice
        board[row, col] = 0

        stats["backtracks"] += 1

    return False


# -----------------------------------------
# 5. Evaluate test dataset
# -----------------------------------------

results = []

print("Starting DQN-Guided Hybrid Backtracking...")
print(f"Total puzzles: {len(df)}")

for index, row in df.iterrows():

    puzzle = np.array(
        [int(x) for x in str(row["quizzes"])],
        dtype=np.int32
    ).reshape(9, 9)

    solution = np.array(
        [int(x) for x in str(row["solutions"])],
        dtype=np.int32
    ).reshape(9, 9)

    original_board = puzzle.copy()
    board = puzzle.copy()

    stats = {
        "nodes": 0,
        "backtracks": 0
    }

    start_time = time.perf_counter()

    solved = solve_sudoku(board, stats)

    elapsed_time = time.perf_counter() - start_time

    # Check against the known correct solution
    correct_solution = (
        solved and np.array_equal(board, solution)
    )

    # Accuracy among originally empty cells
    empty_mask = original_board == 0
    empty_count = int(np.sum(empty_mask))

    if solved and empty_count > 0:
        correct_cells = np.sum(
            (board == solution) & empty_mask
        )
        accuracy = correct_cells / empty_count * 100
    else:
        accuracy = 100.0 if solved and empty_count == 0 else 0.0

    level = {
        5: 1,
        15: 2,
        25: 3,
        35: 4,
        45: 5
    }.get(empty_count, 0)

    results.append({
        "Puzzle": index + 1,
        "Level": level,
        "Empty Cells": empty_count,
        "Solved": int(correct_solution),
        "Cell Accuracy": accuracy,
        "Nodes": stats["nodes"],
        "Backtracks": stats["backtracks"],
        "Time Seconds": elapsed_time
    })

    if (index + 1) % 10 == 0:
        print(f"Evaluated {index + 1}/{len(df)}")


# -----------------------------------------
# 6. Save results
# -----------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)


# -----------------------------------------
# 7. Overall evaluation
# -----------------------------------------

total = len(results_df)
solved_count = results_df["Solved"].sum()

solve_rate = solved_count / total * 100

average_accuracy = results_df["Cell Accuracy"].mean()
average_nodes = results_df["Nodes"].mean()
average_backtracks = results_df["Backtracks"].mean()
average_time = results_df["Time Seconds"].mean()

print("\n========== OVERALL RESULTS ==========")
print(f"Total puzzles: {total}")
print(f"Solved puzzles: {solved_count}")
print(f"Solve Rate: {solve_rate:.2f}%")
print(f"Average Cell Accuracy: {average_accuracy:.2f}%")
print(f"Average Search Nodes: {average_nodes:.2f}")
print(f"Average Backtracks: {average_backtracks:.2f}")
print(f"Average Time: {average_time:.6f} seconds")
print("=====================================")


# -----------------------------------------
# 8. Results per difficulty level
# -----------------------------------------

print("\n========== RESULTS PER LEVEL ==========")

for level in sorted(results_df["Level"].unique()):

    if level == 0:
        continue

    level_df = results_df[
        results_df["Level"] == level
    ]

    level_total = len(level_df)
    level_solved = level_df["Solved"].sum()

    print(f"\nLevel {level}")
    print(f"Solved: {level_solved}/{level_total}")
    print(
        f"Solve Rate: "
        f"{level_solved / level_total * 100:.2f}%"
    )
    print(
        f"Average Nodes: "
        f"{level_df['Nodes'].mean():.2f}"
    )
    print(
        f"Average Backtracks: "
        f"{level_df['Backtracks'].mean():.2f}"
    )
    print(
        f"Average Time: "
        f"{level_df['Time Seconds'].mean():.6f} seconds"
    )

print("\nDetailed results saved to:")
print(OUTPUT_PATH)
