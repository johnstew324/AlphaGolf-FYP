"""
Configuration file for Golf Tournament DQN model
Edit this file to customize model parameters
"""

# Data configuration
DATA_CONFIG = {
    # Percentage of missing values above which columns will be dropped
    'missing_threshold': 30,
    
    # Minimum variance threshold for features
    'variance_threshold': 0.01,
    
    # Percentage of tournaments to use for holdout set
    'holdout_pct': 0.2,
    
    # Percentage of remaining tournaments to use for test set
    'test_pct': 0.2,
    
    # Random seed for reproducibility
    'random_seed': 42
}

# DQN Agent configuration
AGENT_CONFIG = {
    # Network architecture
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.2,
    
    # Exploration parameters
    'epsilon_start': 1.0,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.995,
    
    # Learning parameters
    'learning_rate': 0.001,
    'gamma': 0.95,  # Discount factor
    
    # Memory and batch configuration
    'memory_size': 5000,  # Replay buffer size
    'batch_size': 32,
    
    # Action space size (0: skip, 1: select as winner)
    'action_size': 2
}

# Training configuration
TRAINING_CONFIG = {
    # Number of training episodes
    'num_episodes': 100,
    
    # How often to save model checkpoints
    'checkpoint_interval': 10,
    
    # Default output directory
    'output_dir': './outputs',
    
    # Reward system
    'rewards': {
        'correct_winner': 10,
        'incorrect_winner': -5,
        'skip_winner': -1,
        'indecision_penalty': -2
    }
}

# Hardware optimization for AMD RX580 and 16GB RAM
HARDWARE_CONFIG = {
    # Whether to enable memory growth for GPU
    'memory_growth': True,
    
    # Limit on tournaments to process at once for evaluation
    'tournament_batch_size': 20,
    
    # Whether to use mixed precision
    'mixed_precision': True,
    
    # Number of parallel jobs for hyperparameter tuning
    # None = automatic detection based on CPU cores
    'parallel_jobs': None
}

# Hyperparameter tuning configuration
TUNING_CONFIG = {
    # Tuning method: 'grid' or 'random'
    'method': 'random',
    
    # Number of iterations for random search
    'num_iterations': 20,
    
    # Sample size (number of tournaments) for quicker tuning
    'sample_size': 20,
    
    # Number of episodes for each parameter evaluation
    'eval_episodes': 5,
    
    # Parameter space to search
    'param_grid': {
        'learning_rate': [0.01, 0.005, 0.001, 0.0005],
        'batch_size': [16, 32, 64],
        'hidden_layers': [[128, 64], [128, 64, 32], [64, 32]],
        'gamma': [0.9, 0.95, 0.99],
        'epsilon_decay': [0.99, 0.995, 0.9975],
        'dropout_rate': [0.1, 0.2, 0.3]
    }
}