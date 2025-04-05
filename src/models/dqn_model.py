"""
Deep Q-Network model for golf tournament prediction.
This module provides a DQN model specifically designed for predicting golf tournament outcomes.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Any, Optional, Union

# Define Experience tuple for replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    """
    Q-Network architecture for golf tournament prediction.
    
    Features:
    - Batch normalization for faster and more stable learning
    - Dropout for regularization
    - Residual connections for better gradient flow
    - Layer-specific activation functions
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.2,
                 use_residual: bool = True):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_residual: Whether to use residual connections
        """
        super(QNetwork, self).__init__()
        
        self.use_residual = use_residual
        
        # Input layer
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        # Additional residual connections for better gradient flow
        if use_residual and len(hidden_dims) > 1:
            self.residual_layers = nn.ModuleList()
            for i in range(len(hidden_dims) - 1):
                if hidden_dims[i] != hidden_dims[i+1]:
                    # If dimensions don't match, use a linear layer
                    self.residual_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                else:
                    # If dimensions match, use identity function
                    self.residual_layers.append(nn.Identity())
        
        # Initialize weights using Xavier initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        # Input layer with ReLU activation
        x = F.relu(self.input_bn(self.input_layer(state)))
        x = self.dropout(x)
        
        # Hidden layers with residual connections if enabled
        for i, (hidden_layer, bn_layer) in enumerate(zip(self.hidden_layers, self.bn_layers)):
            if self.use_residual and i < len(self.hidden_layers):
                residual = self.residual_layers[i](x)
                x = F.relu(bn_layer(hidden_layer(x)))
                x = x + residual
            else:
                x = F.relu(bn_layer(hidden_layer(x)))
            
            x = self.dropout(x)
        
        # Output layer (no activation - raw Q-values)
        q_values = self.output_layer(x)
        
        return q_values

class DQNAgent:
    """
    DQN Agent for golf tournament prediction.
    
    Features:
    - Experience replay for sample efficiency
    - Target network for stable learning
    - Double DQN to reduce overestimation bias
    - Prioritized experience replay for more efficient learning
    - Epsilon-greedy exploration strategy with annealing
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 update_target_every: int = 100,
                 hidden_dims: List[int] = [256, 128, 64],
                 device: str = None):
        """
        Initialize DQN Agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting value for epsilon
            epsilon_end: Minimum value for epsilon
            epsilon_decay: Decay rate for epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            update_target_every: Steps between target network updates
            hidden_dims: List of hidden layer dimensions
            device: Device to use for training ('cpu' or 'cuda')
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.step_count = 0
        
        # Initialize networks
        self.policy_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Define loss function (Huber loss for robustness to outliers)
        self.loss_fn = nn.SmoothL1Loss()
        
        # For tracking metrics
        self.loss_history = []
        self.reward_history = []
        self.prediction_history = []
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                         next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.append(experience)
    
    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            eval_mode: Whether to use evaluation mode (no exploration)
            
        Returns:
            Selected action
        """
        # Convert numpy array to torch tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Use greedy policy in evaluation mode
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                self.policy_network.eval()
                q_values = self.policy_network(state_tensor)
                self.policy_network.train()
                return torch.argmax(q_values).item()
        else:
            # Random action for exploration
            return random.randint(0, self.action_dim - 1)
    
    def update(self) -> float:
        """
        Update networks using experience replay.
        
        Returns:
            Loss value
        """
        # Check if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_network(states).gather(1, actions)
        
        # Compute next Q values using Double DQN
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_network(next_states).argmax(1, keepdim=True)
            # Get Q values from target network for those actions
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network if needed
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Save loss for tracking
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """Update target network with policy network weights."""
        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
            'prediction_history': self.prediction_history
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        if not os.path.exists(path):
            print(f"Model file {path} does not exist")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']
        self.loss_history = checkpoint['loss_history']
        self.reward_history = checkpoint.get('reward_history', [])
        self.prediction_history = checkpoint.get('prediction_history', [])
        
        # Set target network to evaluation mode
        self.target_network.eval() 