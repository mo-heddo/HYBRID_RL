
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import numpy as np
import pandas as pd

from training.curriculum import SudokuCurriculum


df = pd.read_csv(
    "dataset/sudoku.csv",
    nrows=1000
)

puzzles = np.array([
    np.array(list(map(int, p))).reshape(9, 9)
    for p in df["quizzes"]
])

solutions = np.array([
    np.array(list(map(int, s))).reshape(9, 9)
    for s in df["solutions"]
])

curriculum = SudokuCurriculum(puzzles, solutions)

for level in range(5):

    level_puzzles, level_solutions = curriculum.get_puzzles()

    print(f"\nLevel {level + 1}")
    print("Number of puzzles:", len(level_puzzles))

    if len(level_puzzles) > 0:

        print(
            "Average empty cells:",
            np.mean(np.sum(level_puzzles == 0, axis=(1, 2)))
        )

    if level < 4:
        curriculum.advance_level()
