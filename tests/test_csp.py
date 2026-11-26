
import pandas as pd
import numpy as np

from algorithms.csp import SudokuCSP


# Load one puzzle from the dataset
df = pd.read_csv("dataset/sudoku.csv")

puzzle = df.iloc[0]["quizzes"]

board = np.array(
    list(map(int, puzzle))
).reshape(9, 9)


# Initialize CSP
csp = SudokuCSP(board)


# Detect empty cells
empty_cells = csp.get_empty_cells()

print("Original board:")
print(board)

print("\nNumber of empty cells:", len(empty_cells))

# Get candidates for all empty cells
candidates = csp.get_all_candidates()

print("\nCandidates:")
for cell, values in list(candidates.items())[:10]:
    print(f"Cell {cell}: {values}")

print("\nContradiction:", csp.has_contradiction())
