
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.csp import SudokuCSP
from strategies.mrv import MRVStrategy


# --------------------------------
# 1. Load test dataset
# --------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "dataset" / "test_curriculum.csv"
OUTPUT_PATH = ROOT / "evaluation" / "csp_mrv_curriculum_results.csv"

df = pd.read_csv(DATA_PATH)

results = []

mrv = MRVStrategy()


# --------------------------------
# 2. CSP + MRV Backtracking Solver
# --------------------------------

def solve_sudoku(board):
    """
    Solve Sudoku using CSP + MRV + Backtracking.
    Returns True if solved, False otherwise.
    """

    csp = SudokuCSP(board)

    # Select the empty cell with the fewest candidates
    cell, candidates = mrv.select_cell(board)

    # No empty cells: puzzle solved
    if cell is None:
        return True

    # No legal candidates: backtrack
    if len(candidates) == 0:
        return False

    row, col = cell

    # Try every legal candidate
    for value in candidates:

        board[row, col] = value

        if solve_sudoku(board):
            return True

        # Undo incorrect choice
        board[row, col] = 0

    return False


# --------------------------------
# 3. Evaluate all test puzzles
# --------------------------------

total = len(df)

print("Starting CSP + MRV evaluation...")
print(f"Total test puzzles: {total}")

for index, row in df.iterrows():

    puzzle = np.array(
        [int(x) for x in str(row["quizzes"])],
        dtype=np.int32
    ).reshape(9, 9)

    solution = np.array(
        [int(x) for x in str(row["solutions"])],
        dtype=np.int32
    ).reshape(9, 9)

    original_puzzle = puzzle.copy()

    solved = solve_sudoku(puzzle.copy())

    # Check whether the returned board matches the known solution
    correct_solution = solved and np.array_equal(puzzle, solution)

    # The solver modifies the board passed to it.
    # Solve again on a copy to preserve the result.
    test_board = original_puzzle.copy()
    solved = solve_sudoku(test_board)

    correct_solution = solved and np.array_equal(test_board, solution)

    # Cell accuracy among originally empty cells
    empty_mask = original_puzzle == 0
    total_empty = np.sum(empty_mask)

    if total_empty > 0 and solved:
        correct_cells = np.sum(
            (test_board == solution) & empty_mask
        )
        accuracy = (correct_cells / total_empty) * 100
    else:
        accuracy = 100.0 if total_empty == 0 and solved else 0.0

    empty_count = int(total_empty)

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
        "Cell Accuracy": accuracy
    })

    if (index + 1) % 10 == 0:
        print(f"Evaluated {index + 1}/{total}")


# --------------------------------
# 4. Save detailed results
# --------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)


# --------------------------------
# 5. Overall results
# --------------------------------

solved_count = results_df["Solved"].sum()
solve_rate = (solved_count / total) * 100

average_accuracy = results_df["Cell Accuracy"].mean()

print("\n========== OVERALL RESULTS ==========")
print(f"Total puzzles: {total}")
print(f"Solved puzzles: {solved_count}")
print(f"Solve Rate: {solve_rate:.2f}%")
print(f"Average Cell Accuracy: {average_accuracy:.2f}%")
print("=====================================")


# --------------------------------
# 6. Results per difficulty level
# --------------------------------

print("\n========== RESULTS PER LEVEL ==========")

for level in sorted(results_df["Level"].unique()):

    if level == 0:
        continue

    level_df = results_df[results_df["Level"] == level]

    level_total = len(level_df)
    level_solved = level_df["Solved"].sum()

    level_solve_rate = (level_solved / level_total) * 100
    level_accuracy = level_df["Cell Accuracy"].mean()

    print(f"\nLevel {level}")
    print(f"Total puzzles: {level_total}")
    print(f"Solved: {level_solved}/{level_total}")
    print(f"Solve Rate: {level_solve_rate:.2f}%")
    print(f"Average Cell Accuracy: {level_accuracy:.2f}%")

print("\nDetailed results saved to:")
print(OUTPUT_PATH)
