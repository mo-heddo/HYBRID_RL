
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from enviroment.sudoku_env import SudokuEnv
from algorithms.dqn_agent import DQNAgent
from training.curriculum import SudokuCurriculum


# ---------------------------------------
# 1. Load Generated Curriculum Dataset
# ---------------------------------------

df = pd.read_csv("dataset/generated_curriculum.csv")

puzzles = np.array([
    np.array(list(map(int, p))).reshape(9, 9)
    for p in df["quizzes"]
])

solutions = np.array([
    np.array(list(map(int, s))).reshape(9, 9)
    for s in df["solutions"]
])

print("Total generated puzzles:", len(puzzles))


# ---------------------------------------
# 2. Create Curriculum
# ---------------------------------------

curriculum = SudokuCurriculum(puzzles, solutions)


# ---------------------------------------
# 3. Create Environment
# ---------------------------------------

# Start with Level 1 puzzles
level_puzzles, level_solutions = curriculum.get_puzzles()

env = SudokuEnv(
    puzzles=level_puzzles,
    solutions=level_solutions,
    max_steps=100
)


# ---------------------------------------
# 4. Create DQN Agent
# ---------------------------------------

agent = DQNAgent(
    state_size=83,
    action_size=9,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    batch_size=64,
    buffer_capacity=10000,
    target_update=100
)


# ---------------------------------------
# 5. Convert Sudoku Board to State
# ---------------------------------------

def encode_state(board, current_cell):

    board_features = board.flatten() / 9.0

    if current_cell is None:
        row, col = 0, 0
    else:
        row, col = current_cell

    row = row / 8.0
    col = col / 8.0

    state = np.concatenate([
        board_features,
        [row, col]
    ])

    return state.astype(np.float32)


# ---------------------------------------
# 6. Training Configuration
# ---------------------------------------

episodes_per_level = 200
total_episodes = episodes_per_level * len(curriculum.levels)

episode_rewards = []
episode_losses = []
training_history = []


# ---------------------------------------
# 7. Curriculum Training Loop
# ---------------------------------------

for level_index in range(len(curriculum.levels)):

    # Get puzzles for current level
    level_puzzles, level_solutions = curriculum.get_puzzles()

    # Update environment dataset
    env.puzzles = level_puzzles
    env.solutions = level_solutions

    current_level = curriculum.get_current_level()

    print("\n===================================")
    print(f"Starting Curriculum Level {current_level}")
    print(f"Empty cells: {curriculum.levels[level_index]}")
    print(f"Available puzzles: {len(level_puzzles)}")
    print("===================================\n")

    for episode in range(episodes_per_level):

        observation, info = env.reset()

        state = encode_state(
            observation,
            info["current_cell"]
        )

        action_mask = info["action_mask"]

        total_reward = 0
        losses = []

        terminated = False
        truncated = False

        while not (terminated or truncated):

            # Select action
            action = agent.select_action(
                state,
                action_mask
            )

            # Execute action
            next_observation, reward, terminated, truncated, next_info = (
                env.step(action)
            )

            # Encode next state
            next_state = encode_state(
                next_observation,
                next_info["current_cell"]
            )

            next_action_mask = next_info["action_mask"]

            # Store experience
            agent.remember(
                state,
                action,
                reward,
                next_state,
                terminated or truncated,
                next_action_mask
            )

            # Train DQN
            loss = agent.train_step()

            if loss is not None:
                losses.append(loss)

            # Update state
            state = next_state
            action_mask = next_action_mask

            total_reward += reward

        # Decay exploration
        agent.decay_epsilon()

        # Record results
        episode_rewards.append(total_reward)

        average_loss = (
            np.mean(losses) if losses else 0.0
        )

        episode_losses.append(average_loss)

        training_history.append({
            "level": current_level,
            "episode": len(episode_rewards),
            "reward": total_reward,
            "loss": average_loss,
            "epsilon": agent.epsilon
        })

        # Print progress
        if (episode + 1) % 10 == 0:

            avg_reward = np.mean(episode_rewards[-10:])

            print(
                f"Level: {current_level}/5 | "
                f"Episode: {episode + 1}/{episodes_per_level} | "
                f"Reward: {total_reward:.2f} | "
                f"Average Reward: {avg_reward:.2f} | "
                f"Loss: {average_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    # Move to next curriculum level
    if not curriculum.is_last_level():
        curriculum.advance_level()


# ---------------------------------------
# 8. Save Trained Model and History
# ---------------------------------------

Path("models").mkdir(exist_ok=True)

agent.save("models/sudoku_dqn_curriculum.pth")

history_df = pd.DataFrame(training_history)

history_df.to_csv(
    "training/curriculum_training_history.csv",
    index=False
)

print("\n===================================")
print("Curriculum Training Completed!")
print("Model saved to models/sudoku_dqn_curriculum.pth")
print("Training history saved.")
print("===================================")
