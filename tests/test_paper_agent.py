import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from algorithms.paper_dqn_agent import PaperDQNAgent
import numpy as np

agent = PaperDQNAgent()

board = np.zeros((9, 9), dtype=np.float32)
occupied_mask = np.zeros(729, dtype=np.int8)

action = agent.select_action(board, occupied_mask)

print("Device:", agent.device)
print("Selected action:", action)
print("Action range:", 0 <= action < 729)