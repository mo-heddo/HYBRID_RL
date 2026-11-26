
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithms.paper_dqn_agent import PaperDQNAgent
from enviroment.paper_sudoku_env import PaperSudokuEnv


# ==============================
# Configuration
# ==============================

DATA_PATH = ROOT / "dataset" / "sudoku.csv"
MODEL_PATH = ROOT / "models" / "sudoku_paper_dqn.pth"
HISTORY_PATH = ROOT / "training" / "paper_training_history.csv"

NUM_EPISODES = 1000
MAX_STEPS = 100

LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 256
BUFFER_SIZE = 100000
TARGET_UPDATE = 1000

SEED = 42


# ==============================
# Reproducibility
# ==============================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")


# ==============================
# Load dataset
# ==============================

df = pd.read_csv(
    DATA_PATH,
    dtype={"quizzes": str, "solutions": str},
    nrows=100
)

puzzles = []
solutions = []

for _, row in df.iterrows():

    puzzle = np.array(
        [int(x) for x in row["quizzes"]],
        dtype=np.int32
    ).reshape(9, 9)

    solution = np.array(
        [int(x) for x in row["solutions"]],
        dtype=np.int32
    ).reshape(9, 9)

    puzzles.append(puzzle)
    solutions.append(solution)

puzzles = np.array(puzzles)
solutions = np.array(solutions)

print("Training puzzles:", len(puzzles))


# ==============================
# Create environment and agent
# ==============================

env = PaperSudokuEnv(
    puzzles=puzzles,
    solutions=solutions,
    max_steps=MAX_STEPS
)

agent = PaperDQNAgent(
    state_size=81,
    action_size=729,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    target_update=TARGET_UPDATE
)


# ==============================
# Training
# ==============================

history = []

successful_episodes = 0

for episode in range(1, NUM_EPISODES + 1):

    state, info = env.reset()

    total_reward = 0
    losses = []

    done = False

    while not done:

        # Select action using the occupied-cell mask
        action = agent.select_action(
            state,
            info["occupied_mask"]
        )

        if action is None:
            break

        next_state, reward, terminated, truncated, next_info = (
            env.step(action)
        )

        done = terminated or truncated

        # Store transition
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done,
            next_info["occupied_mask"]
        )

        # Train the network
        loss = agent.train_step()

        if loss is not None:
            losses.append(loss)

        state = next_state
        info = next_info

        total_reward += reward

    # A successful episode must solve the actual puzzle
    solved = np.array_equal(env.board, env.solution)

    if solved:
        successful_episodes += 1

    # Reduce exploration
    agent.decay_epsilon()

    avg_loss = (
        float(np.mean(losses))
        if losses else 0.0
    )

    history.append({
        "episode": episode,
        "reward": total_reward,
        "solved": int(solved),
        "loss": avg_loss,
        "epsilon": agent.epsilon
    })

    if episode % 10 == 0:

        recent = history[-10:]

        avg_reward = np.mean(
            [item["reward"] for item in recent]
        )

        recent_success = sum(
            item["solved"] for item in recent
        )

        print(
            f"Episode: {episode}/{NUM_EPISODES} | "
            f"Reward: {total_reward:.2f} | "
            f"Average Reward: {avg_reward:.2f} | "
            f"Solved: {recent_success}/10 | "
            f"Loss: {avg_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )


# ==============================
# Save model and history
# ==============================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

agent.save(MODEL_PATH)

pd.DataFrame(history).to_csv(
    HISTORY_PATH,
    index=False
)

print("\n========== TRAINING COMPLETE ==========")

print("Successful episodes:", successful_episodes)

print(
    "Training success rate:",
    f"{successful_episodes / NUM_EPISODES * 100:.2f}%"
)

print("Model saved to:", MODEL_PATH)
print("History saved to:", HISTORY_PATH)
