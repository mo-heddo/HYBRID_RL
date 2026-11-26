
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import numpy as np
import pandas as pd

from enviroment.paper_sudoku_env import PaperSudokuEnv


df = pd.read_csv(
    "dataset/sudoku.csv",
    nrows=10
)

puzzles = np.array([
    np.array(list(map(int, p))).reshape(9, 9)
    for p in df["quizzes"]
])

solutions = np.array([
    np.array(list(map(int, s))).reshape(9, 9)
    for s in df["solutions"]
])

env = PaperSudokuEnv(puzzles, solutions)

observation, info = env.reset()

print("Board shape:", observation.shape)
print("Action space:", env.action_space.n)
print("Observation shape:", env.observation_space.shape)
print("Occupied mask:", info["occupied_mask"].shape)

# Test a valid action on the first empty cell
empty_cell = np.argwhere(env.board == 0)[0]

row, col = empty_cell

digit = int(env.solution[row, col])

action = (row * 9 + col) * 9 + (digit - 1)

observation, reward, terminated, truncated, info = env.step(action)

print("Test action:", action)
print("Reward:", reward)
print("Placed digit:", env.board[row, col])
print("Terminated:", terminated)

print("Environment test completed!")
