
import numpy as np


class SudokuCurriculum:

    def __init__(self, puzzles, solutions):

        self.puzzles = np.array(puzzles)
        self.solutions = np.array(solutions)

        # Count empty cells in every puzzle
        self.empty_counts = np.sum(
            self.puzzles == 0,
            axis=(1, 2)
        )

        # Exact difficulty levels in generated_curriculum.csv
        self.levels = [5, 15, 25, 35, 45]

        self.current_level = 0

    def get_puzzles(self):

        target_empty = self.levels[self.current_level]

        indices = np.where(
            self.empty_counts == target_empty
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
        else:
            print("Curriculum completed!")

    def get_current_level(self):
        return self.current_level + 1

    def is_last_level(self):
        return self.current_level == len(self.levels) - 1
