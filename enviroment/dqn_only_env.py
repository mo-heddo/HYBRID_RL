
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DQNOnlySudokuEnv(gym.Env):

    def __init__(self, puzzles, solutions, max_steps=100):
        super().__init__()

        self.puzzles = puzzles
        self.solutions = solutions
        self.max_steps = max_steps

        # 81 cells * 9 digits
        self.action_space = spaces.Discrete(729)

        # Flattened Sudoku board
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(81,),
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
        )

        self.solution = np.array(
            self.solutions[index],
            dtype=np.int32
        )

        self.steps = 0

        return self._get_observation(), self._get_info()

    def _get_observation(self):
        return (self.board.flatten() / 9.0).astype(np.float32)

    def _get_info(self):

        # Only prevent selecting already-filled cells.
        # All 9 digits remain possible for every empty cell.
        action_mask = np.zeros(729, dtype=np.int8)

        empty_cells = np.where(self.board.flatten() == 0)[0]

        for cell in empty_cells:
            action_mask[cell * 9:(cell + 1) * 9] = 1

        return {"action_mask": action_mask}

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

    def step(self, action):

        self.steps += 1

        action = int(action)

        cell_index = action // 9
        digit = (action % 9) + 1

        row = cell_index // 9
        col = cell_index % 9

        reward = 0.0
        terminated = False
        truncated = False

        # Reject selecting an occupied cell
        if self.board[row, col] != 0:
            reward = -10.0
            terminated = True

        # Reject Sudoku rule violations
        elif not self._is_valid_move(row, col, digit):
            reward = -10.0
            terminated = True

        else:
            self.board[row, col] = digit

            reward = 1.0

            # Oracle-guided reward, same principle as hybrid training
            if digit == self.solution[row, col]:
                reward += 2.0

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
