"""
Training script for DQN model for golf tournament prediction.
This script trains a DQN model to predict golf tournament outcomes.
"""

import os
import sys
import argparse
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import json
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from models.dqn_model import DQNAgent
from models.golf_tournament_env import GolfTournamentEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("models/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN for golf tournament prediction")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="processed_data/enhanced_dqn_data.csv", 
                        help="Path to the enhanced DQN data")
    parser.add_argument("--output_dir", type=str, default="models/trained", 
                        help="Directory to save trained models")
    parser.add_argument("--target", type=str, default="winner", 
                        choices=["winner", "top10_finish", "made_cut"], 
                        help="Prediction target")
    
    # Training arguments
    parser.add_argument("--episodes", type=int, default=1000, 
                        help="Number of episodes for training")
    parser.add_argument("--eval_frequency", type=int, default=50, 
                        help="Evaluate model every N episodes")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, 
                        help="Size of replay buffer")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    
    # Model arguments
    parser.add_argument("--hidden_dims", type=str, default="256,128,64", 
                        help="Hidden dimensions of the Q-network (comma-separated)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--target_update_freq", type=int, default=100, 
                        help="Target network update frequency")
    
    # Exploration arguments
    parser.add_argument("--epsilon_start", type=float, default=1.0, 
                        help="Starting value for epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, 
                        help="Minimum value for epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, 
                        help="Decay rate for epsilon")
    
    # Device arguments
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cpu, cuda, or None for auto)")
    
    # Tuning arguments
    parser.add_argument("--tune", action="store_true", 
                        help="Run hyperparameter tuning instead of training")
    parser.add_argument("--tune_trials", type=int, default=10, 
                        help="Number of trials for hyperparameter tuning")
    
    return parser.parse_args()

def make_env(args: argparse.Namespace) -> GolfTournamentEnv:
    """
    Create a golf tournament environment.
    
    Args:
        args: Command line arguments
        
    Returns:
        Golf tournament environment
    """
    return GolfTournamentEnv(
        data_path=args.data_path,
        prediction_target=args.target,
        test_mode=False,
        tournament_selection='random',
        reward_scale=1.0
    )

def train_dqn(args: argparse.Namespace) -> Tuple[DQNAgent, Dict[str, List[float]]]:
    """
    Train a DQN agent.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (trained agent, metrics dictionary)
    """
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = make_env(args)
    
    # Create DQN agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.replay_buffer_size,
        batch_size=args.batch_size,
        update_target_every=args.target_update_freq,
        hidden_dims=hidden_dims,
        device=args.device
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'episode_accuracies': [],
        'episode_precisions': [],
        'episode_recalls': [],
        'losses': [],
        'epsilons': []
    }
    
    # Train the agent
    logger.info(f"Starting training for {args.episodes} episodes")
    
    total_steps = 0
    best_accuracy = 0.0
    start_time = time.time()
    
    for episode in range(args.episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Initialize episode metrics
        episode_losses = []
        episode_steps = 0
        
        # Run episode
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            if loss > 0:
                episode_losses.append(loss)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
        
        # Log episode results
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_accuracies'].append(info.get('accuracy', 0.0))
        metrics['episode_precisions'].append(info.get('precision', 0.0))
        metrics['episode_recalls'].append(info.get('recall', 0.0))
        metrics['epsilons'].append(agent.epsilon)
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        metrics['losses'].append(avg_loss)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-10:])
            avg_accuracy = np.mean(metrics['episode_accuracies'][-10:])
            avg_precision = np.mean(metrics['episode_precisions'][-10:])
            avg_recall = np.mean(metrics['episode_recalls'][-10:])
            elapsed_time = time.time() - start_time
            
            logger.info(f"Episode {episode + 1}/{args.episodes} | "
                         f"Steps: {episode_steps} | "
                         f"Reward: {episode_reward:.2f} | "
                         f"Accuracy: {info.get('accuracy', 0.0):.2f} | "
                         f"Epsilon: {agent.epsilon:.2f} | "
                         f"Loss: {avg_loss:.4f} | "
                         f"Time: {elapsed_time:.1f}s")
        
        # Evaluate model
        if (episode + 1) % args.eval_frequency == 0:
            eval_accuracy = evaluate_agent(agent, args, num_episodes=10)
            logger.info(f"Evaluation accuracy: {eval_accuracy:.4f}")
            
            # Save model if it's the best so far
            if eval_accuracy > best_accuracy:
                best_accuracy = eval_accuracy
                model_path = os.path.join(args.output_dir, f"best_dqn_{args.target}.pth")
                agent.save(model_path)
                logger.info(f"Saved best model with accuracy {best_accuracy:.4f} to {model_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f"final_dqn_{args.target}.pth")
    agent.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"metrics_{args.target}.json")
    with open(metrics_path, 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in metrics.items()}, f)
    
    # Plot metrics
    plot_metrics(metrics, args.output_dir, args.target)
    
    return agent, metrics

def evaluate_agent(agent: DQNAgent, args: argparse.Namespace, num_episodes: int = 10) -> float:
    """
    Evaluate a DQN agent.
    
    Args:
        agent: DQN agent to evaluate
        args: Command line arguments
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Average accuracy
    """
    # Create environment
    env = make_env(args)
    env.set_test_mode(True)  # Use test mode for deterministic evaluation
    
    # Initialize metrics
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    
    # Evaluate for num_episodes
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Select action without exploration
            action = agent.act(state, eval_mode=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
        
        # Get metrics
        total_accuracy += info.get('accuracy', 0.0)
        total_precision += info.get('precision', 0.0)
        total_recall += info.get('recall', 0.0)
    
    # Calculate averages
    avg_accuracy = total_accuracy / num_episodes
    avg_precision = total_precision / num_episodes
    avg_recall = total_recall / num_episodes
    
    logger.info(f"Evaluation results over {num_episodes} episodes:")
    logger.info(f"  Accuracy: {avg_accuracy:.4f}")
    logger.info(f"  Precision: {avg_precision:.4f}")
    logger.info(f"  Recall: {avg_recall:.4f}")
    
    return avg_accuracy

def plot_metrics(metrics: Dict[str, List[float]], output_dir: str, target: str) -> None:
    """
    Plot training metrics.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save plots
        target: Prediction target
    """
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot rewards
    axes[0].plot(metrics['episode_rewards'])
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True)
    
    # Plot metrics
    axes[1].plot(metrics['episode_accuracies'], label='Accuracy')
    axes[1].plot(metrics['episode_precisions'], label='Precision')
    axes[1].plot(metrics['episode_recalls'], label='Recall')
    axes[1].set_title('Episode Metrics')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot loss and epsilon
    ax1 = axes[2]
    ax2 = ax1.twinx()
    ax1.plot(metrics['losses'], 'r-', label='Loss')
    ax2.plot(metrics['epsilons'], 'b-', label='Epsilon')
    ax1.set_title('Loss and Epsilon')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Epsilon')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"metrics_{target}.png"))
    plt.close()

def tune_hyperparameters(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run hyperparameter tuning for DQN.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of best hyperparameters
    """
    from sklearn.model_selection import ParameterGrid
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'gamma': [0.9, 0.95, 0.99],
        'epsilon_decay': [0.99, 0.995, 0.999],
        'hidden_dims': [
            [128, 64],
            [256, 128, 64],
            [512, 256, 128]
        ]
    }
    
    # Create grid
    grid = list(ParameterGrid(param_grid))
    logger.info(f"Running hyperparameter tuning with {len(grid)} combinations")
    
    # Limit number of trials if needed
    if len(grid) > args.tune_trials:
        grid = np.random.choice(grid, size=args.tune_trials, replace=False)
        logger.info(f"Limiting to {args.tune_trials} trials")
    
    # Initialize best results
    best_accuracy = 0.0
    best_params = None
    results = []
    
    # Run trials
    for i, params in enumerate(grid):
        # Set parameters
        args.learning_rate = params['learning_rate']
        args.gamma = params['gamma']
        args.epsilon_decay = params['epsilon_decay']
        args.hidden_dims = ','.join(str(dim) for dim in params['hidden_dims'])
        
        # Reduce number of episodes for tuning
        tuning_episodes = 200
        original_episodes = args.episodes
        args.episodes = tuning_episodes
        
        logger.info(f"Trial {i+1}/{len(grid)} with params: {params}")
        
        # Train agent
        try:
            agent, metrics = train_dqn(args)
            
            # Evaluate agent
            accuracy = evaluate_agent(agent, args, num_episodes=10)
            
            # Save results
            result = {
                'params': params,
                'accuracy': accuracy,
                'avg_reward': np.mean(metrics['episode_rewards'][-20:])
            }
            results.append(result)
            
            # Update best params
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                logger.info(f"New best accuracy: {best_accuracy:.4f} with params: {best_params}")
        
        except Exception as e:
            logger.error(f"Error in trial {i+1}: {e}")
        
        # Restore original episodes
        args.episodes = original_episodes
    
    # Log best results
    logger.info(f"Hyperparameter tuning complete")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Save results
    results_path = os.path.join(args.output_dir, f"tuning_results_{args.target}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_accuracy': float(best_accuracy),
            'all_results': [
                {
                    'params': {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in r['params'].items()},
                    'accuracy': float(r['accuracy']),
                    'avg_reward': float(r['avg_reward'])
                }
                for r in results
            ]
        }, f)
    
    logger.info(f"Saved tuning results to {results_path}")
    
    return best_params

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args for reproducibility
    args_path = os.path.join(args.output_dir, f"args_{args.target}.json")
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Arguments saved to {args_path}")
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file {args.data_path} not found")
        return
    
    # Run hyperparameter tuning if requested
    if args.tune:
        logger.info("Running hyperparameter tuning")
        best_params = tune_hyperparameters(args)
        
        # Update args with best parameters
        args.learning_rate = best_params['learning_rate']
        args.gamma = best_params['gamma']
        args.epsilon_decay = best_params['epsilon_decay']
        args.hidden_dims = ','.join(str(dim) for dim in best_params['hidden_dims'])
        
        logger.info(f"Training with best parameters: {best_params}")
    
    # Train model
    logger.info(f"Training DQN for {args.target} prediction")
    agent, metrics = train_dqn(args)
    
    # Evaluate final model
    logger.info("Evaluating final model")
    accuracy = evaluate_agent(agent, args, num_episodes=20)
    logger.info(f"Final model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 