from golf_tournament_dqn import DQNAgent

# Initialize a test agent
agent = DQNAgent(
    state_size=len(feature_list),
    action_size=2,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    learning_rate=0.001,
    batch_size=32
)

# Verify model structure
agent.model.summary()