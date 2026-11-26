
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import random

from dataset.generate_curriculum import generate_puzzle

random.seed(123)

# Skip the first 20 original puzzles used for training.
df = pd.read_csv(
    "dataset/sudoku.csv",
    skiprows=range(1, 21),
    nrows=20
)

levels = {
    "Level 1": 5,
    "Level 2": 15,
    "Level 3": 25,
    "Level 4": 35,
    "Level 5": 45
}

generated = []

for index, row in df.iterrows():

    solution = np.array(
        [int(x) for x in row["solutions"]],
        dtype=np.int32
    ).reshape(9, 9)

    for level, target_empty in levels.items():

        puzzle = generate_puzzle(solution, target_empty)

        if puzzle is not None:

            generated.append({
                "quizzes": puzzle,
                "solutions": row["solutions"],
                "empty_cells": target_empty,
                "level": level
            })

            print(f"Generated {level} from test solution {index}")

        else:
            print(f"Could not generate {level} from test solution {index}")

output = pd.DataFrame(generated)

output.to_csv(
    "dataset/test_curriculum.csv",
    index=False
)

print("\nTest dataset generated!")
print("Total puzzles:", len(output))

if not output.empty:
    print(output["level"].value_counts().sort_index())
