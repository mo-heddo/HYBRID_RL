
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/sudoku.csv")

empty_counts = df["quizzes"].apply(
    lambda puzzle: puzzle.count("0")
)

print("Total puzzles:", len(df))
print("\nEmpty cells distribution:")
print(empty_counts.describe())

print("\nFrequency of empty cells:")
print(empty_counts.value_counts().sort_index())

print("\nDifficulty ranges:")

ranges = [
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-30", 21, 30),
    ("31-40", 31, 40),
    ("41-50", 41, 50),
    ("51-60", 51, 60),
    ("61-81", 61, 81)
]

for name, low, high in ranges:

    count = ((empty_counts >= low) &
             (empty_counts <= high)).sum()

    print(f"{name}: {count}")
