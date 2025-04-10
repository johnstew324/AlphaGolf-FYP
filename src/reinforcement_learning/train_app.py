import pandas as pd
import numpy as np
from golf_tournament_dqn import data_preparation, DQNAgent, train_dqn_agent

# Prepare data
train_df, test_df, holdout_df, feature_list = data_preparation("features_set_with_winners.csv")

# Print the first few features to verify
print("First 10 features:", feature_list[:10])

# Verify that position_numeric is NOT in the feature list (it should be in target columns)
if 'position_numeric' in feature_list:
    print("WARNING: position_numeric is in feature list! Removing it...")
    feature_list.remove('position_numeric')


features_to_remove = []
for feature in feature_list:
    for df in [train_df, test_df, holdout_df]:
        if feature in df.columns:
            if df[feature].dtype == 'object':
                print(f"Removing string feature: {feature}, sample: {df[feature].dropna().head(3).tolist()}")
                features_to_remove.append(feature)
                break

# Remove problematic features
for feature in features_to_remove:
    if feature in feature_list:
        feature_list.remove(feature)

print(f"Cleaned feature count: {len(feature_list)}")

# Save cleaned feature list
with open("feature_list.txt", "w") as f:
    for feature in feature_list:
        f.write(f"{feature}\n")

# Train model
print("Starting training...")
agent, history = train_dqn_agent(
    train_df=train_df,
    test_df=test_df,
    feature_list=feature_list,
    num_episodes=100,
    batch_size=64,
    learning_rate=0.001,
    model_dir="./model",
    auto_tune=False
)

# Save processed data for Streamlit app
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
holdout_df.to_csv("holdout_data.csv", index=False)

print("Training complete and data saved!")