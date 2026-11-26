
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PaperSudokuEnv(gym.Env):

    def __init__(self, puzzles, solutions, max_steps=100):

        super().__init__()

        self.puzzles = puzzles
        self.solutions = solutions
        self.max_steps = max_steps

        # 81 cells x 9 possible digits
        self.action_space = spaces.Discrete(729)

        self.observation_space = spaces.Box(
            low=0,
            high=9,
            shape=(9, 9),
            dtype=np.float32
        )

        self.board = None
        self.solution = None
        self.steps = 0

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        index = self.np_random.integers(len(self.puzzles))

        self.board = np.array(
            self.puzzles[index],
            dtype=np.int32
        ).copy()

        self.solution = np.array(
            self.solutions[index],
            dtype=np.int32
        ).copy()

        self.steps = 0

        return self._get_observation(), self._get_info()

    def _get_observation(self):

        return self.board.astype(np.float32)

    def _get_info(self):

        # 0 = available cell, 1 = occupied cell
        occupied_mask = np.repeat(
            (self.board.flatten() != 0),
            9
        ).astype(np.int8)

        return {
            "occupied_mask": occupied_mask
        }

    def _is_valid_move(self, row, col, digit):

        if digit in self.board[row, :]:
            return False

        if digit in self.board[:, col]:
            return False

        box_row = (row // 3) * 3
        box_col = (col // 3) * 3

        box = self.board[
            box_row:box_row + 3,
            box_col:box_col + 3
        ]

        if digit in box:
            return False

        return True

    def _completed_units(self, row, col):

        reward = 0

        # Completed row
        if set(self.board[row, :]) == set(range(1, 10)):
            reward += 10

        # Completed column
        if set(self.board[:, col]) == set(range(1, 10)):
            reward += 10

        # Completed 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3

        box = self.board[
            box_row:box_row + 3,
            box_col:box_col + 3
        ]

        if set(box.flatten()) == set(range(1, 10)):
            reward += 10

        return reward

    def step(self, action):

        self.steps += 1

        action = int(action)

        cell_index = action // 9
        digit = (action % 9) + 1

        row = cell_index // 9
        col = cell_index % 9

        reward = 1.0
        terminated = False
        truncated = False

        # Occupied cell
        if self.board[row, col] != 0:

            reward = -5.0

        # Invalid Sudoku move
        elif not self._is_valid_move(row, col, digit):

            reward = -10.0
            terminated = True

        else:

            # Place digit
            self.board[row, col] = digit

            # Reward completed row, column and box
            reward += self._completed_units(row, col)

            # Completed puzzle
            if np.array_equal(self.board, self.solution):

                reward += 100.0
                terminated = True

            elif not np.any(self.board == 0):

                terminated = True

        if self.steps >= self.max_steps and not terminated:

            truncated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
