
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/generated_curriculum.csv")

print("Total puzzles:", len(df))

print("\nPuzzles per level:")
print(df["level"].value_counts().sort_index())

print("\nChecking puzzle lengths and empty cells...")

for index, row in df.iterrows():

    puzzle = row["quizzes"]
    solution = row["solutions"]

    assert len(puzzle) == 81
    assert len(solution) == 81

    assert puzzle.count("0") == row["empty_cells"]

    # Every given number must match the solution.
    for i in range(81):
        if puzzle[i] != "0":
            assert puzzle[i] == solution[i]

print("All structural checks passed!")

print("\nChecking solution uniqueness...")

from sys import path
from pathlib import Path
path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.generate_curriculum import count_solutions

for index, row in df.iterrows():

    board = np.array(
        [int(x) for x in row["quizzes"]],
        dtype=np.int32
    ).reshape(9, 9)

    solutions_count = count_solutions(board)

    assert solutions_count == 1, (
        f"Puzzle {index} has {solutions_count} solutions!"
    )

print("All puzzles have exactly one solution!")
