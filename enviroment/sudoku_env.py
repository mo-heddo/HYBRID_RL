
import gymnasium as gym
import numpy as np

from gymnasium import spaces
from algorithms.csp import SudokuCSP
from strategies.mrv import MRVStrategy


class SudokuEnv(gym.Env):

    def __init__(self, puzzles, solutions, max_steps=100):

        super().__init__()

        self.puzzles = puzzles
        self.solutions = solutions

        self.max_steps = max_steps

        self.mrv = MRVStrategy()

        # DQN chooses one digit from 1 to 9
        self.action_space = spaces.Discrete(9)

        # Observation: 9x9 Sudoku board
        self.observation_space = spaces.Box(
            low=0,
            high=9,
            shape=(9, 9),
            dtype=np.int32
        )

        self.board = None
        self.solution = None
        self.initial_board = None

        self.current_cell = None
        self.candidates = []

        self.steps = 0

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Select a random puzzle
        index = self.np_random.integers(len(self.puzzles))

        self.initial_board = np.array(
            self.puzzles[index], dtype=np.int32
        )

        self.board = self.initial_board.copy()

        # Used for optional training reward guidance only
        self.solution = np.array(
            self.solutions[index], dtype=np.int32
        )

        self.steps = 0

        self._update_current_cell()

        return self.board.copy(), self._get_info()

    def _update_current_cell(self):

        self.current_cell, self.candidates = (
            self.mrv.select_cell(self.board)
        )

    def _get_info(self):

        action_mask = np.zeros(9, dtype=np.int8)

        for digit in self.candidates:
            action_mask[digit - 1] = 1

        return {
            "current_cell": self.current_cell,
            "candidates": self.candidates.copy(),
            "action_mask": action_mask
        }

    def step(self, action):

        self.steps += 1

        reward = 0.0
        terminated = False
        truncated = False

        # No empty cells: puzzle completed
        if self.current_cell is None:
            return (
                self.board.copy(),
                0.0,
                True,
                False,
                self._get_info()
            )

        # Convert action 0-8 into digit 1-9
        digit = int(action) + 1

        row, col = self.current_cell

        # Reject invalid actions
        if digit not in self.candidates:

            reward = -10.0

            truncated = True

            return (
                self.board.copy(),
                reward,
                terminated,
                truncated,
                self._get_info()
            )

        # Place selected digit
        self.board[row, col] = digit

        reward = 1.0

        # Optional solution-based reward guidance
        if digit == self.solution[row, col]:
            reward += 2.0

        # Check whether the puzzle is solved
        if np.array_equal(self.board, self.solution):

            reward += 100.0
            terminated = True

        else:

            # Recalculate CSP candidates and MRV cell
            csp = SudokuCSP(self.board)

            if csp.has_contradiction():

                reward = -10.0
                terminated = True

            else:

                self._update_current_cell()

                if self.current_cell is None:
                    terminated = True

                elif len(self.candidates) == 0:
                    reward = -10.0
                    terminated = True

        # Maximum episode length
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        return (
            self.board.copy(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
