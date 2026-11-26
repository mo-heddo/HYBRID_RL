
import pandas as pd
import numpy as np
import random


# Count Sudoku solutions.
# We stop at 2 because we only need to know
# whether the puzzle has exactly one solution.
def count_solutions(board, limit=2):

    empty_cells = []

    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:

                used = set(board[r])
                used.update(board[i][c] for i in range(9))

                box_r = (r // 3) * 3
                box_c = (c // 3) * 3

                for i in range(box_r, box_r + 3):
                    for j in range(box_c, box_c + 3):
                        used.add(board[i][j])

                candidates = [
                    n for n in range(1, 10)
                    if n not in used
                ]

                if not candidates:
                    return 0

                empty_cells.append((len(candidates), r, c, candidates))

    if not empty_cells:
        return 1

    # MRV: choose the cell with the fewest candidates.
    _, row, col, candidates = min(empty_cells)

    total = 0

    for digit in candidates:
        board[row][col] = digit

        total += count_solutions(board, limit)

        board[row][col] = 0

        if total >= limit:
            return total

    return total


# Generate one puzzle with a target number of empty cells.
def generate_puzzle(solution, target_empty):

    board = np.array(solution, dtype=np.int32).copy()

    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)

    empty_count = 0

    for row, col in cells:

        if empty_count >= target_empty:
            break

        original_value = board[row, col]

        # Temporarily remove this number.
        board[row, col] = 0

        # Keep the removal only if there is exactly one solution.
        if count_solutions(board.copy()) == 1:
            empty_count += 1
        else:
            board[row, col] = original_value

    if empty_count != target_empty:
        return None

    return "".join(map(str, board.flatten()))


def main():

    random.seed(42)

    # Start small: use only 20 original solutions.
    df = pd.read_csv("dataset/sudoku.csv", nrows=20)

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

                print(f"Generated {level} from solution {index}")

            else:
                print(
                    f"Could not generate {level} "
                    f"from solution {index}"
                )

    output = pd.DataFrame(generated)

    output.to_csv(
        "dataset/generated_curriculum.csv",
        index=False
    )

    print("\n========== GENERATION SUMMARY ==========")
    print("Total generated puzzles:", len(output))

    if not output.empty:
        print("\nPuzzles per level:")
        print(output["level"].value_counts().sort_index())

    print("\nSaved to dataset/generated_curriculum.csv")


if __name__ == "__main__":
    main()
