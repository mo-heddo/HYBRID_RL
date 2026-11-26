
import pandas as pd
import numpy as np

from strategies.mrv import MRVStrategy


# Load a real Sudoku puzzle
df = pd.read_csv("dataset/sudoku.csv")

puzzle = df.iloc[0]["quizzes"]

board = np.array(
    list(map(int, puzzle))
).reshape(9, 9)


# Initialize MRV
mrv = MRVStrategy()

cell, candidates = mrv.select_cell(board)


print("Selected cell:", cell)
print("Candidates:", candidates)
print("Number of candidates:", len(candidates))


# Display the selected cell
if cell is not None:
    row, col = cell
    print(f"\nSelected coordinate: Row {row}, Column {col}")
