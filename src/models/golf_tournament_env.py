"""
Golf tournament environment for reinforcement learning.
This module provides an environment for training reinforcement learning agents
to predict golf tournament outcomes.
"""

import os
import sys
import numpy as np
import pandas as pd
import random
import gym
from gym import spaces
from typing import Dict, List, Tuple, Any, Optional, Union

class GolfTournamentEnv(gym.Env):
    """
    Golf tournament environment for reinforcement learning.
    
    This environment is designed to train an agent to predict the outcomes
    of golf tournaments. Each tournament consists of a sequence of players,
    and the agent needs to predict whether each player will achieve a specific
    outcome (win, top-10, make cut, etc.).
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 data_path: str,
                 prediction_target: str = 'winner',
                 test_mode: bool = False,
                 tournament_selection: str = 'random',
                 reward_scale: float = 1.0):
        """
        Initialize the environment.
        
        Args:
            data_path: Path to the dataset file
            prediction_target: Target variable to predict ('winner', 'top10_finish', 'made_cut')
            test_mode: Whether to use test mode (fixed tournament sequence)
            tournament_selection: Method for selecting tournaments ('random', 'sequential')
            reward_scale: Scale factor for rewards
        """
        super(GolfTournamentEnv, self).__init__()
        
        # Load dataset
        self.data = pd.read_csv(data_path)
        
        # Basic validation
        if self.data.empty:
            raise ValueError(f"Dataset at {data_path} is empty")
            
        required_columns = ['tournament_id', 'player_id', 'player_name']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Dataset missing required columns: {required_columns}")
            
        # Set prediction target
        self.prediction_target = prediction_target
        if prediction_target not in self.data.columns:
            raise ValueError(f"Prediction target '{prediction_target}' not in dataset")
        
        # Set parameters
        self.test_mode = test_mode
        self.tournament_selection = tournament_selection
        self.reward_scale = reward_scale
        
        # Get unique tournaments
        self.tournaments = self.data['tournament_id'].unique()
        self.num_tournaments = len(self.tournaments)
        print(f"Environment initialized with {self.num_tournaments} tournaments")
        
        # Set tournament index for sequential selection
        self.current_tournament_idx = 0
        
        # Initialize state tracking
        self.current_tournament = None
        self.players_in_tournament = []
        self.current_player_idx = 0
        self.current_state = None
        
        # Initialize metrics tracking
        self.total_reward = 0
        self.predictions = []
        self.actual_results = []
        self.episode_rewards = []
        
        # Define action and observation spaces
        # Binary action space: 0 (negative prediction) or 1 (positive prediction)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: player features
        self.feature_columns = self.get_feature_columns()
        feature_count = len(self.feature_columns)
        
        # High and low bounds for observation space
        high = np.ones(feature_count)
        low = -np.ones(feature_count)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        print(f"Observation space: {self.observation_space.shape}")
        print(f"Action space: {self.action_space.n}")
    
    def get_feature_columns(self) -> List[str]:
        """
        Get the list of feature columns to use for the state.
        
        Returns:
            List of feature column names
        """
        # Exclude non-feature columns
        exclude_columns = [
            'player_id', 'player_name', 'tournament_id', 'year',
            'reward', 'normalized_reward', 'position', 'winner',
            'made_cut', 'top10_finish', 'top20_finish'
        ]
        
        # Get all numeric columns except excluded ones
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        return feature_columns
    
    def get_player_features(self, player_data: pd.Series) -> np.ndarray:
        """
        Extract features for a given player.
        
        Args:
            player_data: Player data Series
            
        Returns:
            NumPy array of player features
        """
        # Extract features
        features = player_data[self.feature_columns].values.astype(np.float32)
        
        # Check for NaN values and replace with 0
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial state
        """
        # Select tournament
        if self.tournament_selection == 'random':
            # Random tournament
            self.current_tournament_idx = np.random.randint(0, self.num_tournaments)
        elif self.tournament_selection == 'sequential':
            # Sequential tournament (wrap around when reaching the end)
            if self.current_tournament_idx >= self.num_tournaments:
                self.current_tournament_idx = 0
        
        # Get tournament ID
        self.current_tournament = self.tournaments[self.current_tournament_idx]
        
        # Get tournament data
        tournament_data = self.data[self.data['tournament_id'] == self.current_tournament]
        if len(tournament_data) == 0:
            # If no data found, try next tournament
            self.current_tournament_idx += 1
            return self.reset()
        
        # Reset tournament state
        self.players_in_tournament = tournament_data.index.tolist()
        
        # Shuffle players unless in test mode
        if not self.test_mode:
            random.shuffle(self.players_in_tournament)
        
        # Reset player index
        self.current_player_idx = 0
        
        # Get initial state
        player_data = self.data.loc[self.players_in_tournament[self.current_player_idx]]
        self.current_state = self.get_player_features(player_data)
        
        # Reset metrics
        self.total_reward = 0
        self.predictions = []
        self.actual_results = []
        
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0=negative prediction, 1=positive prediction)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if not 0 <= action < self.action_space.n:
            raise ValueError(f"Invalid action: {action}. Must be 0 or 1")
        
        # Get current player data
        player_data = self.data.loc[self.players_in_tournament[self.current_player_idx]]
        
        # Get actual result for the prediction target
        actual_result = player_data[self.prediction_target]
        
        # Calculate reward
        # Positive reward if prediction matches actual result, negative otherwise
        if action == actual_result:
            # Correct prediction
            if actual_result == 1:
                # True positive (correctly predicted positive outcome)
                # Higher reward for correctly predicting rare positive outcomes
                reward = 1.0
            else:
                # True negative (correctly predicted negative outcome)
                reward = 0.1
        else:
            # Incorrect prediction
            if action == 1 and actual_result == 0:
                # False positive
                reward = -0.2
            else:
                # False negative (missed a positive outcome - penalize more)
                reward = -1.0
        
        # Apply reward scaling
        reward *= self.reward_scale
        
        # Update total reward
        self.total_reward += reward
        
        # Store prediction and actual result
        self.predictions.append(action)
        self.actual_results.append(actual_result)
        
        # Move to next player
        self.current_player_idx += 1
        done = self.current_player_idx >= len(self.players_in_tournament)
        
        if done:
            # Episode is finished
            self.episode_rewards.append(self.total_reward)
            self.current_tournament_idx += 1
            
            # Calculate metrics
            accuracy = self.calculate_accuracy()
            precision = self.calculate_precision()
            recall = self.calculate_recall()
            
            info = {
                'episode_reward': self.total_reward,
                'tournament_id': self.current_tournament,
                'num_players': len(self.players_in_tournament),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'predictions': self.predictions,
                'actual_results': self.actual_results
            }
        else:
            # Move to next player
            player_data = self.data.loc[self.players_in_tournament[self.current_player_idx]]
            self.current_state = self.get_player_features(player_data)
            info = {}
        
        return self.current_state, reward, done, info
    
    def calculate_accuracy(self) -> float:
        """
        Calculate prediction accuracy.
        
        Returns:
            Accuracy (0-1)
        """
        if not self.predictions:
            return 0.0
        
        correct = sum(pred == actual for pred, actual in zip(self.predictions, self.actual_results))
        return correct / len(self.predictions)
    
    def calculate_precision(self) -> float:
        """
        Calculate prediction precision.
        
        Returns:
            Precision (0-1)
        """
        # Precision = TP / (TP + FP)
        if not self.predictions:
            return 0.0
        
        true_positives = sum(pred == 1 and actual == 1 for pred, actual in zip(self.predictions, self.actual_results))
        predicted_positives = sum(pred == 1 for pred in self.predictions)
        
        if predicted_positives == 0:
            return 0.0
        
        return true_positives / predicted_positives
    
    def calculate_recall(self) -> float:
        """
        Calculate prediction recall.
        
        Returns:
            Recall (0-1)
        """
        # Recall = TP / (TP + FN)
        if not self.predictions:
            return 0.0
        
        true_positives = sum(pred == 1 and actual == 1 for pred, actual in zip(self.predictions, self.actual_results))
        actual_positives = sum(actual == 1 for actual in self.actual_results)
        
        if actual_positives == 0:
            return 0.0
        
        return true_positives / actual_positives
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            None for 'human' mode
        """
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} not supported")
        
        # Print current tournament and player
        if self.current_tournament is not None and self.current_player_idx < len(self.players_in_tournament):
            player_data = self.data.loc[self.players_in_tournament[self.current_player_idx]]
            player_name = player_data['player_name']
            print(f"Tournament: {self.current_tournament}")
            print(f"Player: {player_name}")
            print(f"Total reward so far: {self.total_reward:.2f}")
            print(f"Accuracy so far: {self.calculate_accuracy():.2f}")
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def get_tournament_by_id(self, tournament_id: str) -> pd.DataFrame:
        """
        Get data for a specific tournament.
        
        Args:
            tournament_id: Tournament ID
            
        Returns:
            DataFrame with tournament data
        """
        return self.data[self.data['tournament_id'] == tournament_id]
    
    def set_test_mode(self, test_mode: bool) -> None:
        """
        Set test mode.
        
        Args:
            test_mode: Whether to use test mode
        """
        self.test_mode = test_mode 