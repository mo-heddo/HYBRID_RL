
import numpy as np


class SudokuCSP:

    def __init__(self, board):
        self.board = np.array(board, dtype=np.int32)

        if self.board.shape != (9, 9):
            raise ValueError("Sudoku board must have shape (9, 9).")

    def get_empty_cells(self):
        """
        Return coordinates of all empty cells.
        """
        return [tuple(cell) for cell in np.argwhere(self.board == 0)]

    def get_candidates(self, row, col):
        """
        Return all valid digits for a given empty cell.
        """

        if self.board[row, col] != 0:
            return []

        # Existing numbers in the row
        row_values = set(self.board[row, :])

        # Existing numbers in the column
        col_values = set(self.board[:, col])

        # Identify the 3x3 box dynamically
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3

        box_values = set(
            self.board[
                box_row:box_row + 3,
                box_col:box_col + 3
            ].flatten()
        )

        # Remove zeros because they represent empty cells
        used_values = row_values | col_values | box_values
        used_values.discard(0)

        return sorted(set(range(1, 10)) - used_values)

    def get_all_candidates(self):
        """
        Return candidates for every empty cell.
        """

        return {
            (row, col): self.get_candidates(row, col)
            for row, col in self.get_empty_cells()
        }

    def is_valid_move(self, row, col, value):
        """
        Check whether a digit is valid for a cell.
        """

        return value in self.get_candidates(row, col)

    def has_contradiction(self):
        """
        Check whether an empty cell has no valid candidates.
        """

        return any(
            len(candidates) == 0
            for candidates in self.get_all_candidates().values()
        )
