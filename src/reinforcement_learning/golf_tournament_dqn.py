import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
import os
import pickle
from datetime import datetime
import argparse

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Hardware optimization for AMD GPU
# Try to optimize for AMD RX580 with 16GB RAM and Ryzen 3600X
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Using GPU: {physical_devices}")
    except:
        print("Unable to configure GPU memory growth. Using default settings.")

# If no GPU is found or configuration fails, optimize for CPU (Ryzen 3600X)
if not physical_devices:
    print("No GPU found. Using CPU optimization for Ryzen 3600X")
    # Use moderate batch sizes for CPU training

class ReplayBuffer:
    """Replay Buffer to store experiences for DQN training"""
    
    def __init__(self, max_size=5000):  # Reduced buffer size for 16GB RAM
        self.buffer = deque(maxlen=max_size)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        if buffer_size < batch_size:
            return []
            
        index = np.random.choice(np.arange(buffer_size), 
                                 size=batch_size, 
                                 replace=False)
        
        return [self.buffer[i] for i in index]
    
    def size(self):
        return len(self.buffer)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
            
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.buffer = pickle.load(f)

class DQNAgent:
    """Deep Q-Network Agent for Golf Tournament Winner Prediction"""
    
    def __init__(
        self, 
        state_size, 
        action_size=2, 
        epsilon=1.0, 
        epsilon_min=0.05, 
        epsilon_decay=0.995, 
        learning_rate=0.001, 
        gamma=0.95, 
        batch_size=32,  # Moderate batch size for RX580
        memory_size=5000,  # Reduced memory size for 16GB RAM
        hidden_layers=[128, 64, 32],
        dropout_rate=0.2
    ):
        """
        Initialize the DQN Agent
        
        Parameters:
        -----------
        state_size : int
            Size of the input state (feature vector)
        action_size : int
            Number of possible actions (default: 2 - skip or select)
        epsilon : float
            Initial exploration rate
        epsilon_min : float
            Minimum exploration rate
        epsilon_decay : float
            Rate at which epsilon decreases
        learning_rate : float
            Learning rate for model updates
        gamma : float
            Discount factor for future rewards
        batch_size : int
            Number of samples to use for each training step
        memory_size : int
            Maximum size of replay memory
        hidden_layers : list
            List of hidden layer sizes
        dropout_rate : float
            Dropout rate for regularization
        """
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Create replay memory
        self.memory = ReplayBuffer(max_size=memory_size)
        
        # Create model
        self.model = self._build_model()
        
        # Metrics
        self.episode_rewards = []
        self.training_accuracy = []
        self.training_loss = []
    
    def _build_model(self):
        """Build neural network model for DQN"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], input_dim=self.state_size, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer - Q-values for each action
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(
            loss='mse', 
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.add((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action based on epsilon-greedy policy"""
        try:
            # Debug state information
            if isinstance(state, np.ndarray) and state.dtype == 'object':
                print(f"WARNING: State has object dtype, shape: {state.shape}")
                # Convert to float32
                state = state.astype(np.float32)
            
            if training and np.random.rand() <= self.epsilon:
                # Exploration - random action
                return random.randrange(self.action_size)
            
            # Exploitation - choose best action based on Q-values
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
        except Exception as e:
            print(f"Error in act method: {str(e)}")
            print(f"State type: {type(state)}")
            if isinstance(state, np.ndarray):
                print(f"State shape: {state.shape}, dtype: {state.dtype}")
                if len(state) > 0:
                    print(f"First few values: {state[0][:5] if len(state[0]) > 5 else state[0]}")
            
            # Let's check for non-numeric values
            if isinstance(state, np.ndarray) and state.dtype == 'object':
                print("Examining state values for non-numeric entries:")
                for i, val in enumerate(state.flatten()):
                    if not isinstance(val, (int, float, np.number)):
                        print(f"  Non-numeric value at position {i}: {val} (type: {type(val)})")
                        if i >= 10:  # Limit to first 10 issues
                            print("  ... more issues may exist")
                            break
            
            # Re-raise the exception
            raise
    
    def replay(self):
        """Train model using randomly sampled experiences from replay memory"""
        if self.memory.size() < self.batch_size:
            return 0  # Not enough samples for training
        
        # Sample batch of experiences
        minibatch = self.memory.sample(self.batch_size)
        
        try:
            # Debug info
            print("Checking experience data types...")
            
            states = np.array([experience[0][0] for experience in minibatch])
            actions = np.array([experience[1] for experience in minibatch])
            rewards = np.array([experience[2] for experience in minibatch])
            next_states = np.array([experience[3][0] for experience in minibatch])
            dones = np.array([experience[4] for experience in minibatch])
            
            # Print data type information
            print(f"States shape: {states.shape}, dtype: {states.dtype}")
            print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
            print(f"Rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
            print(f"Next states shape: {next_states.shape}, dtype: {next_states.dtype}")
            print(f"Dones shape: {dones.shape}, dtype: {dones.dtype}")
            
            # Convert any object arrays to float32
            if states.dtype == 'object':
                print("Converting states to float32...")
                states = states.astype(np.float32)
            if next_states.dtype == 'object':
                print("Converting next_states to float32...")
                next_states = next_states.astype(np.float32)
            
            # Current Q-values for all actions
            targets = self.model.predict(states, verbose=0)
            
            # Next state Q-values for target
            next_q_values = self.model.predict(next_states, verbose=0)
            
            # Update target Q-values with reward + gamma * max Q-value for next state
            for i in range(len(minibatch)):
                if dones[i]:
                    # For terminal states, just use the reward
                    targets[i, actions[i]] = rewards[i]
                else:
                    # For non-terminal states, add discounted future rewards
                    targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
            
            # Train the model
            history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
            
            # Decrease exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            return history.history['loss'][0]
        except Exception as e:
            print(f"Error in replay: {str(e)}")
            print("Let's examine one experience to debug:")
            experience = minibatch[0]
            for i, part in enumerate(['state', 'action', 'reward', 'next_state', 'done']):
                print(f"{part}: {type(experience[i])}")
                if i in [0, 3]:  # state or next_state
                    if hasattr(experience[i], 'shape'):
                        print(f"  Shape: {experience[i].shape}")
                    if hasattr(experience[i], 'dtype'):
                        print(f"  dtype: {experience[i].dtype}")
                    print(f"  First few values: {experience[i][0][:5] if len(experience[i]) > 0 else None}")
            
            # Raise the exception to stop execution
            raise
    
    def calculate_win_probability(self, tournament_df, features):
        """
        Calculate win probability for each player in a tournament
        
        Parameters:
        -----------
        tournament_df : DataFrame
            Data for a single tournament
        features : list
            List of feature columns to use
        
        Returns:
        --------
        DataFrame
            Original tournament data with added win probability column
        """
        result_df = tournament_df.copy()
        
        # Prepare states for all players
        states = []
        player_indices = []
        
        for idx, row in tournament_df.iterrows():
            state = row[features].values.reshape(1, -1)
            states.append(state)
            player_indices.append(idx)
        
        # Get Q-values for all players
        q_values = []
        for state in states:
            q_value = self.model.predict(state, verbose=0)[0][1]  # Q-value for 'select as winner' action
            q_values.append(q_value)
        
        # Calculate win probabilities using softmax
        q_values = np.array(q_values)
        
        # Scale Q-values to avoid numerical issues with softmax
        scaled_q_values = q_values - np.max(q_values)
        win_probs = np.exp(scaled_q_values) / np.sum(np.exp(scaled_q_values))
        
        # Add win probabilities to the result dataframe
        result_df['win_probability'] = 0.0
        for i, idx in enumerate(player_indices):
            result_df.loc[idx, 'win_probability'] = win_probs[i]
        
        # Add raw Q-values
        result_df['q_value'] = 0.0
        for i, idx in enumerate(player_indices):
            result_df.loc[idx, 'q_value'] = q_values[i]
        
        return result_df
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
    def save_memory(self, filepath):
        """Save the replay memory"""
        self.memory.save(filepath)
        print(f"Replay memory saved to {filepath}")
        
    def load_memory(self, filepath):
        """Load replay memory"""
        self.memory.load(filepath)
        print(f"Replay memory loaded from {filepath}")

def data_preparation(filepath):
    """
    Prepare data for DQN model
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing golf tournament data
    
    Returns:
    --------
    tuple
        (train_df, test_df, holdout_df, feature_list)
    """
    print("Starting data preparation...")
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Basic info
    print(f"Number of tournaments: {df['tournament_id'].nunique()}")
    print(f"Number of players: {df['player_id'].nunique()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    high_missing_cols = missing_pct[missing_pct > 30].index.tolist()
    
    print(f"\nDropping {len(high_missing_cols)} columns with >30% missing values")
    df = df.drop(columns=high_missing_cols)
    
    # Impute remaining missing values
    columns_with_missing = df.columns[df.isnull().any()].tolist()
    for col in columns_with_missing:
        if col not in ['player_id', 'tournament_id']:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    
    # Check if tournament_id is string format like "R2025016"
    if not pd.api.types.is_numeric_dtype(df['tournament_id']):
        print("Tournament ID is in string format, converting to categorical code")
        df['tournament_id_code'] = df['tournament_id'].astype('category').cat.codes
    else:
        df['tournament_id_code'] = df['tournament_id']
    
    # Normalize any remaining unscaled features
    # Exclude ID columns and columns that appear already scaled
    scaled_pattern = ['_scaled', '_norm', '_pct', '_ratio', 'percentage']
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    to_scale = []
    for col in numerical_cols:
        # Skip ID columns and columns that appear already scaled
        if col in ['player_id', 'tournament_id', 'tournament_id_code'] or any(pattern in col.lower() for pattern in scaled_pattern):
            continue
            
        # Check data range for normalization decision
        col_max = df[col].max()
        col_min = df[col].min()
        
        if col_max > 10 or col_min < -10:
            to_scale.append(col)
    
    if to_scale:
        print(f"Normalizing {len(to_scale)} unscaled numerical features")
        scaler = StandardScaler()
        df[to_scale] = scaler.fit_transform(df[to_scale])
    
    # Identify the winner in each tournament
    # This implementation depends on your data structure
    
    # Option 1: If you have a direct indicator column
    if any(col in df.columns for col in ['is_winner', 'winner']):
        winner_col = 'is_winner' if 'is_winner' in df.columns else 'winner'
        print(f"Using existing winner column: {winner_col}")
    else:
        # Option 2: If winner is deducible from other columns like standings or position
        potential_rank_cols = ['standings_rank_2_scaled']
        rank_col = next((col for col in potential_rank_cols if col in df.columns), None)
        
        if rank_col:
            print(f"Using {rank_col} to determine winners")
            # Identify the player with lowest (best) rank in each tournament
            winners = df.loc[df.groupby('tournament_id_code')[rank_col].idxmin()]
            df['is_winner'] = 0
            df.loc[winners.index, 'is_winner'] = 1
        else:
            # Option 3: If winner isn't clear, create synthetic winners based on available metrics
            print("Creating synthetic winners using performance metrics")
            
            # Use features like strokes gained total, scoring average, etc.
            metrics = []
            
            if 'strokes_gained_scoring_sg_total_total_sg:t' in df.columns:
                metrics.append('strokes_gained_scoring_sg_total_total_sg:t')
            
            if 'scoring_scoring_average_actual' in df.columns:
                # Lower is better for scoring average, so we negate it
                df['neg_scoring_average'] = -df['scoring_scoring_average_actual']
                metrics.append('neg_scoring_average')
            
            # Create composite score from available metrics
            if metrics:
                # Normalize each metric for equal weighting
                for metric in metrics:
                    df[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                
                # Create composite score
                norm_cols = [f'{metric}_norm' for metric in metrics]
                df['perf_score'] = df[norm_cols].sum(axis=1)
                
                # Identify winner as player with highest score in each tournament
                winners = df.loc[df.groupby('tournament_id_code')['perf_score'].idxmax()]
                df['is_winner'] = 0
                df.loc[winners.index, 'is_winner'] = 1
                
                print(f"Created synthetic winners using metrics: {metrics}")
            else:
                # If no suitable metrics, use 'top10' if available
                if 'top10' in df.columns or 'top_10' in df.columns:
                    top10_col = 'top10' if 'top10' in df.columns else 'top_10'
                    top10_players = df[df[top10_col] == 1]
                    
                    # Randomly select one top 10 player from each tournament as winner
                    tournaments = top10_players['tournament_id_code'].unique()
                    df['is_winner'] = 0
                    
                    for tourn in tournaments:
                        tourn_players = top10_players[top10_players['tournament_id_code'] == tourn]
                        if not tourn_players.empty:
                            winner_idx = np.random.choice(tourn_players.index)
                            df.loc[winner_idx, 'is_winner'] = 1
                    
                    print(f"Created winners by randomly selecting from top 10 players")
                else:
                    print("Warning: No suitable way to determine winners found")
                    df['is_winner'] = 0
    
    # Check winner counts
    winner_count = df['is_winner'].sum()
    tournament_count = df['tournament_id'].nunique()
    print(f"Identified {winner_count} winners across {tournament_count} tournaments")
    
    # Calculate win percentage for each player (historical data)
    player_stats = df.groupby('player_id').agg(
        tournaments=('tournament_id', 'nunique'),
        wins=('is_winner', 'sum')
    )
    player_stats['win_percentage'] = (player_stats['wins'] / player_stats['tournaments']) * 100
    df = df.merge(player_stats[['win_percentage']], on='player_id', how='left')
    
    # Split data by tournaments
    # Get unique tournament IDs
    tournaments = df['tournament_id_code'].unique()
    
    # Split for holdout set (20%)
    train_test_tournaments, holdout_tournaments = train_test_split(
        tournaments, test_size=0.2, random_state=42
    )
    
    # Further split into train and test (80/20 of remaining data)
    train_tournaments, test_tournaments = train_test_split(
        train_test_tournaments, test_size=0.2, random_state=42
    )
    
    # Create dataframes for each split
    train_df = df[df['tournament_id_code'].isin(train_tournaments)]
    test_df = df[df['tournament_id_code'].isin(test_tournaments)]
    holdout_df = df[df['tournament_id_code'].isin(holdout_tournaments)]
    
    print(f"Data split complete:")
    print(f"Training: {len(train_tournaments)} tournaments, {len(train_df)} rows")
    print(f"Testing: {len(test_tournaments)} tournaments, {len(test_df)} rows")
    print(f"Holdout: {len(holdout_tournaments)} tournaments, {len(holdout_df)} rows")
    
    # Create feature list (exclude non-feature columns)
    exclude_cols = ['player_id', 'tournament_id', 'tournament_id_code', 'is_winner', 
                    'win_percentage', 'perf_score']
    if 'neg_scoring_average' in df.columns:
        exclude_cols.append('neg_scoring_average')
    
    # Also exclude any derived normalization columns we created
    for col in df.columns:
        if col.endswith('_norm') and col not in exclude_cols:
            exclude_cols.append(col)
    
    feature_list = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Selected {len(feature_list)} features for the model")
    
    return train_df, test_df, holdout_df, feature_list

def automated_hyperparameter_search(train_df, test_df, feature_list):
    """
    Perform an automated hyperparameter search for the DQN model
    
    Parameters:
    -----------
    train_df : DataFrame
        Training data
    test_df : DataFrame
        Testing data
    feature_list : list
        List of features to use
    
    Returns:
    --------
    dict
        Best hyperparameters
    """
    print("Starting hyperparameter search...")
    
    # Define parameter ranges based on RX580 GPU and 16GB RAM constraints
    param_grid = {
        'learning_rate': [0.01, 0.005, 0.001, 0.0005],
        'batch_size': [16, 32, 64],  # Keep batch sizes reasonable for RX580
        'hidden_layers': [[128, 64], [128, 64, 32], [64, 32]],  # Model complexity options
        'gamma': [0.9, 0.95, 0.99],
        'epsilon_decay': [0.99, 0.995, 0.9975]
    }
    
    # Number of tournaments to use for quick evaluation
    n_tournaments = min(20, train_df['tournament_id_code'].nunique())
    sample_tournaments = np.random.choice(
        train_df['tournament_id_code'].unique(), 
        size=n_tournaments, 
        replace=False
    )
    sample_df = train_df[train_df['tournament_id_code'].isin(sample_tournaments)]
    
    print(f"Using {n_tournaments} tournaments with {len(sample_df)} rows for hyperparameter tuning")
    
    # Track results
    results = []
    
    # Get feature dimensions
    state_size = len(feature_list)
    
    # For each parameter combination
    total_combinations = (len(param_grid['learning_rate']) * 
                         len(param_grid['batch_size']) * 
                         len(param_grid['hidden_layers']) * 
                         len(param_grid['gamma']) * 
                         len(param_grid['epsilon_decay']))
    
    print(f"Total parameter combinations to evaluate: {total_combinations}")
    print("Using simplified grid search to find best parameters...")
    
    # Simplify the search space for faster results
    # Focus on key parameters that most affect performance
    simplified_grid = {
        'learning_rate': [0.005, 0.001],
        'batch_size': [32],  # Fixed for RX580
        'hidden_layers': [[128, 64]],  # Fixed for simplicity
        'gamma': [0.95],  # Fixed for simplicity
        'epsilon_decay': [0.995]  # Fixed for simplicity
    }
    
    total_simplified = (len(simplified_grid['learning_rate']) * 
                        len(simplified_grid['batch_size']) * 
                        len(simplified_grid['hidden_layers']) * 
                        len(simplified_grid['gamma']) * 
                        len(simplified_grid['epsilon_decay']))
    
    print(f"Simplified to {total_simplified} combinations")
    
    best_reward = -float('inf')
    best_params = None
    
    combination_count = 0
    
    # For each parameter combination in simplified grid
    for lr in simplified_grid['learning_rate']:
        for bs in simplified_grid['batch_size']:
            for hl in simplified_grid['hidden_layers']:
                for gm in simplified_grid['gamma']:
                    for ed in simplified_grid['epsilon_decay']:
                        combination_count += 1
                        print(f"Evaluating combination {combination_count}/{total_simplified}...")
                        
                        # Initialize agent with these parameters
                        agent = DQNAgent(
                            state_size=state_size,
                            learning_rate=lr,
                            batch_size=bs,
                            hidden_layers=hl,
                            gamma=gm,
                            epsilon_decay=ed,
                            memory_size=5000,  # Fixed for 16GB RAM
                            epsilon=1.0,  # Starting with full exploration
                            epsilon_min=0.1
                        )
                        
                        # Quick training for 5 episodes
                        total_reward = 0
                        correct_pred = 0
                        total_tournaments = 0
                        
                        # Training loop
                        for episode in range(5):  # Limited episodes for quick evaluation
                            ep_reward = 0
                            ep_correct = 0
                            ep_tournaments = 0
                            
                            # Get unique tournaments
                            tuning_tournaments = sample_df['tournament_id_code'].unique()
                            np.random.shuffle(tuning_tournaments)
                            
                            # Limit number of tournaments for speed
                            for t_id in tuning_tournaments[:10]:  # Only use 10 tournaments per episode
                                ep_tournaments += 1
                                total_tournaments += 1
                                
                                # Get data for this tournament
                                t_data = sample_df[sample_df['tournament_id_code'] == t_id].copy()
                                t_data = t_data.sample(frac=1).reset_index(drop=True)  # Shuffle
                                
                                player_selected = False
                                
                                # Loop through each player in the tournament
                                for i in range(len(t_data)):
                                    player = t_data.iloc[i]
                                    state = player[feature_list].values.reshape(1, -1)
                                    
                                    if state.dtype == 'object':
                                        print(f"State created with object dtype, shape: {state.shape}")
                                        print(f"Feature list: {feature_list}")
                                        print(f"First few features: {feature_list[:5]}")
                                        print(f"Sample values for first few features:")
                                        for feat in feature_list[:5]:
                                            print(f"  {feat}: {player[feat]} (type: {type(player[feat])})")
                                        
                                        # Try to identify the problematic feature
                                        for feat in feature_list:
                                            val = player[feat]
                                            if not isinstance(val, (int, float, np.number)) or pd.isna(val):
                                                print(f"Potential problematic feature: {feat}, value: {val}, type: {type(val)}")
                                    
                                    # Choose action
                                    action = agent.act(state)
                                    
                                    # If selected as winner
                                    if action == 1:
                                        player_selected = True
                                        
                                        # Check if correct
                                        if player['is_winner'] == 1:
                                            reward = 10
                                            ep_correct += 1
                                            ep_reward += reward
                                        else:
                                            reward = -5
                                            ep_reward += reward
                                        
                                        # Store experience
                                        agent.remember(state, action, reward, state, True)
                                        break
                                    else:  # Skip player
                                        if player['is_winner'] == 1:
                                            reward = -1
                                            ep_reward += reward
                                        
                                        # Store experience if not last player
                                        if i < len(t_data) - 1:
                                            next_player = t_data.iloc[i+1]
                                            next_state = next_player[feature_list].values.reshape(1, -1)
                                            agent.remember(state, action, 0, next_state, False)
                                
                                # Force selection of last player if none selected
                                if not player_selected:
                                    last_player = t_data.iloc[-1]
                                    state = last_player[feature_list].values.reshape(1, -1)
                                    
                                    # Add forced selection
                                    if last_player['is_winner'] == 1:
                                        reward = 10
                                        ep_correct += 1
                                    else:
                                        reward = -5
                                    
                                    ep_reward += reward
                                    agent.remember(state, 1, reward, state, True)
                                
                                # Train after each tournament if enough samples
                                if agent.memory.size() >= agent.batch_size:
                                    agent.replay()
                            
                            # Add episode metrics
                            total_reward += ep_reward
                            correct_pred += ep_correct
                        
                        # Calculate average reward and accuracy
                        avg_reward = total_reward / total_tournaments if total_tournaments > 0 else 0
                        accuracy = correct_pred / total_tournaments if total_tournaments > 0 else 0
                        
                        print(f"Parameters: lr={lr}, bs={bs}, layers={hl}, gamma={gm}, eps_decay={ed}")
                        print(f"Avg Reward: {avg_reward:.2f}, Accuracy: {accuracy:.2f}")
                        
                        # Track results
                        results.append({
                            'learning_rate': lr,
                            'batch_size': bs,
                            'hidden_layers': hl,
                            'gamma': gm,
                            'epsilon_decay': ed,
                            'avg_reward': avg_reward,
                            'accuracy': accuracy
                        })
                        
                        # Update best parameters
                        if avg_reward > best_reward:
                            best_reward = avg_reward
                            best_params = {
                                'learning_rate': lr,
                                'batch_size': bs,
                                'hidden_layers': hl,
                                'gamma': gm,
                                'epsilon_decay': ed
                            }
    
    # Print best parameters
    print("\nBest hyperparameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best average reward: {best_reward:.2f}")
    
    # Return best parameters
    return best_params

def train_dqn_agent(train_df, test_df, feature_list, 
                   num_episodes=100,
                   batch_size=32, 
                   learning_rate=0.001, 
                   gamma=0.95,
                   epsilon_decay=0.995, 
                   hidden_layers=[128, 64, 32],
                   dropout_rate=0.2,
                   memory_size=5000,
                   model_dir="./models",
                   checkpoint_interval=10,
                   auto_tune=False):
    """
    Train a DQN agent for golf tournament winner prediction
    
    Parameters:
    -----------
    train_df : DataFrame
        Training data
    test_df : DataFrame
        Testing data
    feature_list : list
        List of features to use
    num_episodes : int
        Number of training episodes
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for model updates
    gamma : float
        Discount factor for future rewards
    epsilon_decay : float
        Rate at which epsilon decreases
    hidden_layers : list
        List of hidden layer sizes
    dropout_rate : float
        Dropout rate for regularization
    memory_size : int
        Maximum size of replay memory
    model_dir : str
        Directory to save model files
    checkpoint_interval : int
        How often to save model checkpoints
    auto_tune : bool
        Whether to perform hyperparameter tuning
    
    Returns:
    --------
    DQNAgent
        Trained agent
    dict
        Training history and metrics
    """
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Record start time
    start_time = datetime.now()
    print(f"Training started at: {start_time}")
    
    # Get state size
    state_size = len(feature_list)
    print(f"Input feature count: {state_size}")
    
    # Perform hyperparameter tuning if requested
    if auto_tune:
        print("Performing automated hyperparameter tuning...")
        best_params = automated_hyperparameter_search(train_df, test_df, feature_list)
        
        # Update parameters with best values
        learning_rate = best_params['learning_rate']
        batch_size = best_params['batch_size']
        hidden_layers = best_params['hidden_layers']
        gamma = best_params['gamma']
        epsilon_decay = best_params['epsilon_decay']
        
        print("Using tuned hyperparameters for training")
    
    # Initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=2,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        memory_size=memory_size,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate
    )
    
    # Get unique tournaments
    tournament_ids = train_df['tournament_id_code'].unique()
    print(f"Total training tournaments: {len(tournament_ids)}")
    
    # Initialize metrics tracking
    history = {
        'episode_rewards': [],
        'episode_accuracy': [],
        'episode_loss': [],
        'epsilon_values': [],
        'time_per_episode': []
    }
    
    # Main training loop
    print(f"\nStarting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        episode_start = datetime.now()
        
        # Shuffle tournaments for each episode
        np.random.shuffle(tournament_ids)
        
        # Track episode metrics
        total_reward = 0
        correct_predictions = 0
        total_tournaments = 0
        total_loss = 0
        batch_count = 0
        
        # Process each tournament
        for tournament_id in tournament_ids:
            total_tournaments += 1
            
            # Get data for this tournament
            tournament_data = train_df[train_df['tournament_id_code'] == tournament_id].copy()
            
            # Shuffle players to prevent order bias
            tournament_data = tournament_data.sample(frac=1).reset_index(drop=True)
            
            # Track if a player has been selected
            player_selected = False
            
            # Process each player
            for i in range(len(tournament_data)):
                # Current player
                player = tournament_data.iloc[i]
                
                # Create state (feature vector)
                state_data = player[feature_list].fillna(0).values  # Fill NAs with 0
                state = state_data.astype(np.float32).reshape(1, -1)  # Force convert to float32
                
                # Choose action (0: skip, 1: select as winner)
                action = agent.act(state)
                
                # Initialize reward
                reward = 0
                
                # Check if this is the last player
                is_last_player = (i == len(tournament_data) - 1)
                
                # If action is to select this player as winner
                if action == 1:
                    player_selected = True
                    
                    # Check if this is actually the winner
                    if player['is_winner'] == 1:
                        reward = 10  # Correct prediction
                        correct_predictions += 1
                    else:
                        reward = -5  # Incorrect prediction
                    
                    # Store experience in replay memory
                    agent.remember(state, action, reward, state, True)
                    
                    # Add to total reward
                    total_reward += reward
                    
                    # Break out of player loop since we've made a selection
                    break
                    
                else:  # Action is to skip this player
                    # Check if this player was the winner
                    if player['is_winner'] == 1:
                        reward = -1  # Small negative reward for skipping the winner
                        total_reward += reward
                    
                    # If this is the last player and no selection made, force selection
                    if is_last_player and not player_selected:
                        # Force selection of last player
                        action = 1
                        
                        # Check if last player is the winner
                        if player['is_winner'] == 1:
                            reward = 10  # Correct prediction
                            correct_predictions += 1
                        else:
                            reward = -5  # Incorrect prediction
                        
                        # Store experience
                        agent.remember(state, action, reward, state, True)
                        
                        # Add to total reward
                        total_reward += reward
                    else:
                        # Regular skip - only store if not the last player
                        if not is_last_player:
                            next_player = tournament_data.iloc[i+1]
                            next_state = next_player[feature_list].values.reshape(1, -1)
                            agent.remember(state, action, reward, next_state, False)
            
            # Train the model after each tournament if enough samples
            if agent.memory.size() >= agent.batch_size:
                loss = agent.replay()
                total_loss += loss
                batch_count += 1
        
        # Calculate episode metrics
        avg_reward = total_reward / total_tournaments if total_tournaments > 0 else 0
        avg_accuracy = correct_predictions / total_tournaments if total_tournaments > 0 else 0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        # Track history
        history['episode_rewards'].append(avg_reward)
        history['episode_accuracy'].append(avg_accuracy)
        history['episode_loss'].append(avg_loss)
        history['epsilon_values'].append(agent.epsilon)
        
        # Calculate episode duration
        episode_duration = datetime.now() - episode_start
        history['time_per_episode'].append(episode_duration.total_seconds())
        
        # Print progress
        if (episode + 1) % 5 == 0 or episode == 0:
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Reward: {avg_reward:.2f}, Accuracy: {avg_accuracy:.2f}, "
                  f"Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}, "
                  f"Time: {episode_duration}")
        
        # Save checkpoint models periodically
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"dqn_golf_ep{episode+1}.h5")
            agent.save_model(checkpoint_path)
            
            # Also save memory periodically to enable training continuation
            memory_path = os.path.join(model_dir, f"memory_ep{episode+1}.pkl")
            agent.save_memory(memory_path)
    
    # Save final model
    final_model_path = os.path.join(model_dir, "dqn_golf_final.h5")
    agent.save_model(final_model_path)
    
    # Save final memory
    final_memory_path = os.path.join(model_dir, "memory_final.pkl")
    agent.save_memory(final_memory_path)
    
    # Calculate total training time
    total_training_time = datetime.now() - start_time
    print(f"\nTotal training time: {total_training_time}")
    
    # Visualize training metrics
    plot_training_history(history, model_dir)
    
    return agent, history

def plot_training_history(history, save_dir):
    """
    Plot training metrics
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training metrics
    save_dir : str
        Directory to save plots
    """
    plt.figure(figsize=(15, 12))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(history['episode_rewards'])
    plt.title('Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['episode_accuracy'])
    plt.title('Prediction Accuracy per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 2, 3)
    plt.plot(history['episode_loss'])
    plt.title('Training Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    
    # Plot epsilon
    plt.subplot(2, 2, 4)
    plt.plot(history['epsilon_values'])
    plt.title('Exploration Rate (Epsilon) per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    print(f"Training metrics plot saved to {save_dir}/training_metrics.png")
    
    # Optional: Plot time per episode
    plt.figure(figsize=(10, 5))
    plt.plot(history['time_per_episode'])
    plt.title('Time per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'time_per_episode.png'))
    
    plt.close('all')  # Close all figures

def run_prediction(filepath=None, model_path=None, output_dir='./predictions'):
    """
    Main function to run the complete prediction pipeline
    
    Parameters:
    -----------
    filepath : str
        Path to the dataset file
    model_path : str
        Path to a pre-trained model (if None, train a new model)
    output_dir : str
        Directory to save prediction results and models
    
    Returns:
    --------
    None
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Data preparation
    print("Step 1: Data Preparation")
    train_df, test_df, holdout_df, feature_list = data_preparation(filepath)
    
    # Save feature list for future reference
    with open(os.path.join(output_dir, 'feature_list.pkl'), 'wb') as f:
        pickle.dump(feature_list, f)
    
    # Step 2: Model training or loading
    print("\nStep 2: Model Training/Loading")
    if model_path:
        # Load pre-trained model
        state_size = len(feature_list)
        agent = DQNAgent(state_size=state_size)
        agent.load_model(model_path)
        print(f"Loaded pre-trained model from {model_path}")
    else:
        # Train a new model
        print("Training a new model...")
        agent, history = train_dqn_agent(
            train_df=train_df,
            test_df=test_df,
            feature_list=feature_list,
            num_episodes=100,  # Adjust based on available time and resources
            batch_size=32,
            learning_rate=0.001,
            gamma=0.95,
            epsilon_decay=0.995,
            hidden_layers=[128, 64, 32],
            model_dir=output_dir,
            auto_tune=True  # Enable automated hyperparameter tuning
        )
    
    # Step 3: Evaluate on test set
    print("\nStep 3: Model Evaluation on Test Set")
    test_metrics, test_predictions = evaluate_dqn_agent(agent, test_df, feature_list)
    
    # Save test predictions
    test_predictions.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    # Step 4: Final evaluation on holdout set
    print("\nStep 4: Final Evaluation on Holdout Set")
    holdout_metrics, holdout_predictions = evaluate_dqn_agent(agent, holdout_df, feature_list)
    
    # Save holdout predictions
    holdout_predictions.to_csv(os.path.join(output_dir, 'holdout_predictions.csv'), index=False)
    
    # Save metrics
    metrics = {
        'test': test_metrics,
        'holdout': holdout_metrics
    }
    
    with open(os.path.join(output_dir, 'evaluation_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"\nAll results saved to {output_dir}")
    print("Prediction pipeline completed successfully!")

def evaluate_dqn_agent(agent, test_df, feature_list, tournament_batch_size=None):
    """
    Evaluate the trained DQN agent on test data
    
    Parameters:
    -----------
    agent : DQNAgent
        Trained DQN agent
    test_df : DataFrame
        Test data
    feature_list : list
        List of features used by the model
    tournament_batch_size : int
        Number of tournaments to process at once (for memory management)
        If None, process all tournaments at once
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    DataFrame
        Detailed prediction results
    """
    print("\nEvaluating DQN agent on test data...")
    
    # Get unique tournament IDs
    tournament_ids = test_df['tournament_id_code'].unique()
    
    # Initialize metrics
    metrics = {
        'accuracy': 0,
        'top3_accuracy': 0,
        'top5_accuracy': 0,
        'top10_accuracy': 0,
        'avg_winner_probability': 0,
        'avg_winner_rank': 0
    }
    
    # Store detailed results
    all_predictions = []
    
    # Process tournaments in batches if specified
    if tournament_batch_size:
        tournament_batches = [tournament_ids[i:i+tournament_batch_size] 
                             for i in range(0, len(tournament_ids), tournament_batch_size)]
        print(f"Processing {len(tournament_ids)} tournaments in {len(tournament_batches)} batches")
    else:
        tournament_batches = [tournament_ids]
    
    # Process each batch
    for batch_idx, batch_tournaments in enumerate(tournament_batches):
        if tournament_batch_size:
            print(f"Processing batch {batch_idx+1}/{len(tournament_batches)}...")
        
        # Process each tournament in the batch
        for tournament_id in batch_tournaments:
            # Get data for this tournament
            tournament_data = test_df[test_df['tournament_id_code'] == tournament_id].copy()
            
            # Calculate win probabilities
            predictions = agent.calculate_win_probability(tournament_data, feature_list)
            
            # Sort players by predicted win probability
            predictions = predictions.sort_values('win_probability', ascending=False).reset_index(drop=True)
            
            # Find the actual winner
            actual_winner = predictions[predictions['is_winner'] == 1]
            
            if len(actual_winner) > 0:
                # Get winner's rank in predictions
                winner_rank = actual_winner.index[0] + 1  # +1 because index is 0-based
                
                # Get winner's predicted probability
                winner_prob = actual_winner['win_probability'].values[0]
                
                # Update metrics
                metrics['avg_winner_probability'] += winner_prob
                metrics['avg_winner_rank'] += winner_rank
                
                # Check if winner is in top predictions
                if winner_rank == 1:
                    metrics['accuracy'] += 1
                if winner_rank <= 3:
                    metrics['top3_accuracy'] += 1
                if winner_rank <= 5:
                    metrics['top5_accuracy'] += 1
                if winner_rank <= 10:
                    metrics['top10_accuracy'] += 1
            
            # Add tournament ID to predictions
            predictions['tournament_id'] = tournament_id
            
            # Keep key columns for analysis
            keep_cols = ['tournament_id', 'player_id', 'is_winner', 'win_probability', 'q_value']
            additional_cols = [col for col in tournament_data.columns 
                              if col in ['owgr', 'player_name', 'experience_level_numeric', 
                                        'win_percentage', 'tournament_name', 'course_name']]
            keep_cols.extend(additional_cols)
            
            result_cols = [col for col in keep_cols if col in predictions.columns]
            tournament_results = predictions[result_cols].copy()
            
            # Add rank column
            tournament_results['predicted_rank'] = range(1, len(tournament_results) + 1)
            
            # Append to all predictions
            all_predictions.append(tournament_results)
    
    # Combine all tournament predictions
    if all_predictions:
        all_results = pd.concat(all_predictions, ignore_index=True)
    else:
        all_results = pd.DataFrame()
    
    # Calculate average metrics
    total_tournaments = len(tournament_ids)
    if total_tournaments > 0:
        metrics['accuracy'] /= total_tournaments
        metrics['top3_accuracy'] /= total_tournaments
        metrics['top5_accuracy'] /= total_tournaments
        metrics['top10_accuracy'] /= total_tournaments
        metrics['avg_winner_probability'] /= total_tournaments
        metrics['avg_winner_rank'] /= total_tournaments
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy (exact winner): {metrics['accuracy']:.4f} ({metrics['accuracy']*total_tournaments:.0f}/{total_tournaments})")
    print(f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f} ({metrics['top3_accuracy']*total_tournaments:.0f}/{total_tournaments})")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*total_tournaments:.0f}/{total_tournaments})")
    print(f"Top-10 Accuracy: {metrics['top10_accuracy']:.4f} ({metrics['top10_accuracy']*total_tournaments:.0f}/{total_tournaments})")
    print(f"Average Winner Probability: {metrics['avg_winner_probability']:.4f}")
    print(f"Average Winner Rank: {metrics['avg_winner_rank']:.2f}")
    
    return metrics, all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Golf Tournament Winner Prediction with DQN')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--model', type=str, help='Path to pre-trained model (optional)')
    parser.add_argument('--output', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Set up output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"run_{timestamp}")
    
    # Run prediction pipeline
    run_prediction(
        filepath=args.data, 
        model_path=args.model,
        output_dir=output_dir
    )