import pandas as pd
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

from environment.sudoku_env import SudokuEnv


# Load 100 puzzles
df = pd.read_csv("dataset/sudoku.csv").head(100)

puzzles = np.array([
    np.array(list(map(int, p))).reshape(9, 9)
    for p in df["quizzes"]
])

solutions = np.array([
    np.array(list(map(int, s))).reshape(9, 9)
    for s in df["solutions"]
])


# Initialize environment
env = SudokuEnv(puzzles, solutions)

observation, info = env.reset(seed=42)

print("Initial board:")
print(observation)

print("\nSelected cell:", info["current_cell"])
print("Candidates:", info["candidates"])
print("Action mask:", info["action_mask"])

# Choose the first valid candidate for testing
action = info["candidates"][0] - 1

observation, reward, terminated, truncated, info = env.step(action)

print("\nBoard after action:")
print(observation)

print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)