
import numpy as np


class SudokuCurriculum:

    def __init__(self, puzzles, solutions):

        self.puzzles = puzzles
        self.solutions = solutions

        # Count empty cells in every puzzle
        self.empty_counts = np.sum(
            puzzles == 0,
            axis=(1, 2)
        )

        # Five difficulty levels
        self.levels = [
            (1, 1),
            (2, 10),
            (11, 30),
            (31, 50),
            (51, 81)
        ]

        self.current_level = 0

    def get_puzzles(self):

        min_empty, max_empty = self.levels[self.current_level]

        indices = np.where(
            (self.empty_counts >= min_empty) &
            (self.empty_counts <= max_empty)
        )[0]

        return (
            self.puzzles[indices],
            self.solutions[indices]
        )

    def advance_level(self):

        if self.current_level < len(self.levels) - 1:

            self.current_level += 1

        print(
            f"Curriculum advanced to level "
            f"{self.current_level + 1}"
        )
