
from algorithms.csp import SudokuCSP


class MRVStrategy:

    def select_cell(self, board):
        """
        Select the empty cell with the fewest valid candidates.

        Returns:
            cell: (row, col)
            candidates: valid digits for that cell
        """

        csp = SudokuCSP(board)

        all_candidates = csp.get_all_candidates()

        # No empty cells: Sudoku is complete
        if not all_candidates:
            return None, []

        # Find the cell with the minimum number of candidates
        best_cell = min(
            all_candidates,
            key=lambda cell: len(all_candidates[cell])
        )

        return best_cell, all_candidates[best_cell]
