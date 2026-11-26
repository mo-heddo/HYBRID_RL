
import pandas as pd

df = pd.read_csv("training/curriculum_training_history.csv")

# Expected maximum reward for each level
expected_rewards = {
    1: 115,
    2: 145,
    3: 175,
    4: 205,
    5: 235
}

print("\n========== CURRICULUM TRAINING ANALYSIS ==========\n")

for level in sorted(df["level"].unique()):

    level_df = df[df["level"] == level]

    expected = expected_rewards[level]

    successful = (
        level_df["reward"] >= expected
    ).sum()

    total = len(level_df)

    success_rate = (successful / total) * 100

    print(f"Level: {level}")
    print(f"Expected reward: {expected}")
    print(f"Successful episodes: {successful}/{total}")
    print(f"Training success rate: {success_rate:.2f}%")
    print(f"Average reward: {level_df['reward'].mean():.2f}")
    print(f"Average loss: {level_df['loss'].mean():.4f}")
    print(f"Final epsilon: {level_df['epsilon'].iloc[-1]:.4f}")
    print("---------------------------------------")

print("\n==================================================")
