import os
import sys
import logging
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
import time

#DATA_PATH = os.path.join("alphaGolf", "data", "features", "3tournaments.csv")
#DATA_PATH = os.path.join("alphaGolf", "data", "features","model1_tests1", "processed_tournament_data.csv")
#CONFIG_PATH = os.path.join("reinforcement_learning", "config", "model1_config.yaml")
#OUTPUT_DIR = os.path.join("alphaGolf", "data", "model1_training", f"new_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

DATA_PATH = "alphaGolf/data/features/model1_tests1/processed_tournament_data.csv"
#DATA_PATH = "alphaGolf/data/features/processed_tournament_data.csv"  # Path to your processed data
CONFIG_PATH = "reinforcement_learning/config/model1_config.yaml"  # Config path
OUTPUT_DIR = "alphaGolf/data/RL/model1_training/Deep_Learning"  

# Training parameters
NUM_EPISODES = 500  # Number of episodes for full training
TRAINING_STEPS = 500  # Max steps per episode
CHECKPOINT_FREQ = 50  # Save checkpoints every N episodes
EVAL_FREQ = 25  # Evaluate and log metrics every N episodes
EARLY_STOP_PATIENCE = 50  # Early stopping patience (episodes)
MOVING_AVG_WINDOW = 20  # Window size for moving averages

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# ===== IMPORT MODEL COMPONENTS =====
# Try different import paths based on your structure
try:
    print("Trying first import path...")
    from src.reinforcement_learning.environments.basic_environment import GolfTournamentEnv
    from src.reinforcement_learning.agents.dqn.basic_agent import DQNAgent
except ImportError as e:
    print(f"First import path failed: {e}")
    try:
        print("Trying alternative import path...")
        from reinforcement_learning.environments.basic_environment import GolfTournamentEnv
        from reinforcement_learning.agents.dqn.basic_agent import DQNAgent
    except ImportError as e:
        print(f"Alternative import path failed: {e}")
        print("Please adjust import paths to match your directory structure.")
        print(f"Current sys.path: {sys.path}")
        print("Available modules in the current directory:")
        for item in os.listdir('.'):
            print(f"  - {item}")
        sys.exit(1)

print("Imports successful!")

# ===== CREATE CONFIG FILE IF IT DOESN'T EXIST =====
def create_config():
    """Create the configuration file if it doesn't exist"""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    
    config_content = """# Model 1 Configuration - Golf Tournament Prediction

# Environment Configuration
environment:
  reward_type: 'roi'            # 'roi', 'accuracy', or 'combined'
  win_odds: 30.0                # Average odds for tournament winner
  top5_odds: 6.0                # Average odds for top 5 finish
  top10_odds: 3.0               # Average odds for top 10 finish
  top25_odds: 1.5               # Average odds for top 25 finish
  made_cut_odds: 1.2            # Average odds for making the cut
  random_odds_variation: 0.2    # Random variation in odds (±20%)

# Agent Configuration
agent:
  hidden_dims: [128, 64]        # Network architecture
  learning_rate: 0.001          # Learning rate
  gamma: 0.99                   # Discount factor
  tau: 0.001                    # Soft update parameter
  batch_size: 64                # Batch size for training
  buffer_size: 10000            # Replay buffer size
  update_every: 4               # Update frequency
  epsilon_start: 1.0            # Starting exploration rate
  epsilon_end: 0.05             # Minimum exploration rate
  epsilon_decay: 0.995          # Decay rate for exploration
  device: 'cpu'                 # Device to use (cpu or cuda)
"""
    
    with open(CONFIG_PATH, 'w') as f:
        f.write(config_content)
    
    print(f"Created configuration at: {CONFIG_PATH}")

# ===== LOAD CONFIG FILE =====
def load_config():
    """Load configuration from YAML file"""
    try:
        with open(CONFIG_PATH, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Loaded configuration from {CONFIG_PATH}")
        return config
    except FileNotFoundError:
        print(f"Config file not found. Creating default config at {CONFIG_PATH}")
        create_config()
        return load_config()
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {}

# ===== VISUALIZE TRAINING PROGRESS =====
def plot_training_metrics(metrics, output_dir):
    """Create visualizations of training metrics"""
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot episode rewards
    axs[0, 0].plot(metrics['episode'], metrics['reward'])
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    axs[0, 0].grid(True)
    
    # Plot moving average of rewards
    if len(metrics['moving_avg_reward']) > 0:
        axs[0, 1].plot(metrics['episode'][-len(metrics['moving_avg_reward']):], metrics['moving_avg_reward'])
        axs[0, 1].set_title(f'Moving Avg Reward (Window={MOVING_AVG_WINDOW})')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Avg Reward')
        axs[0, 1].grid(True)
    
    # Plot accuracy
    axs[1, 0].plot(metrics['episode'], metrics['accuracy'])
    axs[1, 0].set_title('Prediction Accuracy')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Accuracy (%)')
    axs[1, 0].grid(True)
    
    # Plot epsilon decay
    axs[1, 1].plot(metrics['episode'], metrics['epsilon'])
    axs[1, 1].set_title('Exploration Rate (Epsilon)')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Epsilon')
    axs[1, 1].grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    # If we have loss data, create a separate plot
    if len(metrics['loss']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['loss'])
        plt.title('Training Loss')
        plt.xlabel('Training Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_loss.png'))
        plt.close()
    
    print(f"Saved training visualizations to {output_dir}")

# ===== TRAIN MODEL 1 =====
def train_model1():
    """Run a complete training session for Model 1"""
    print("\n" + "="*60)
    print("  TRAINING MODEL 1")
    print("="*60)
    print(f"Data: {DATA_PATH}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Steps: {TRAINING_STEPS}")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found: {DATA_PATH}")
        return False
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create log file
    log_path = os.path.join(OUTPUT_DIR, 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write(f"Model 1 Training Log\n")
        f.write(f"=================\n\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"Data: {DATA_PATH}\n")
        f.write(f"Episodes: {NUM_EPISODES}\n\n")
    
    # Load configuration
    config = load_config()
    env_config = config.get('environment', {})
    agent_config = config.get('agent', {})
    
    # Save configuration to output directory
    with open(os.path.join(OUTPUT_DIR, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Create environment
    print(f"Creating environment with data from {DATA_PATH}")
    env = GolfTournamentEnv(DATA_PATH, env_config)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create agent
    print("Creating DQN agent")
    agent = DQNAgent(state_dim, action_dim, agent_config)
    
    # Initialize metrics tracking
    metrics = {
        'episode': [],
        'reward': [],
        'accuracy': [],
        'epsilon': [],
        'loss': [],
        'moving_avg_reward': [],
        'moving_avg_accuracy': [],
        'time_elapsed': []
    }
    
    # For early stopping
    best_avg_reward = float('-inf')
    patience_counter = 0
    reward_window = deque(maxlen=MOVING_AVG_WINDOW)
    accuracy_window = deque(maxlen=MOVING_AVG_WINDOW)
    
    # Training loop
    print(f"Running {NUM_EPISODES} training episodes with {TRAINING_STEPS} steps each")
    
    # Create CSV file for detailed metrics
    metrics_path = os.path.join(OUTPUT_DIR, 'detailed_metrics.csv')
    with open(metrics_path, 'w') as f:
        f.write("episode,reward,accuracy,epsilon,time_elapsed,moving_avg_reward,moving_avg_accuracy\n")
    
    for episode in range(1, NUM_EPISODES + 1):
        episode_start_time = time.time()
        
        # Reset environment
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        # Episode loop
        for step in range(TRAINING_STEPS):
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Process step
            agent.step(state, action, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Check if episode is done
            if done:
                break
        
        # Get agent metrics
        agent_metrics = agent.get_metrics()
        
        # Update metrics
        epsilon = agent.epsilon
        accuracy = info.get('accuracy', 0)
        time_elapsed = time.time() - start_time
        
        metrics['episode'].append(episode)
        metrics['reward'].append(episode_reward)
        metrics['accuracy'].append(accuracy)
        metrics['epsilon'].append(epsilon)
        
        # Update moving averages
        reward_window.append(episode_reward)
        accuracy_window.append(accuracy)
        
        if len(reward_window) == MOVING_AVG_WINDOW:
            moving_avg_reward = sum(reward_window) / MOVING_AVG_WINDOW
            moving_avg_accuracy = sum(accuracy_window) / MOVING_AVG_WINDOW
            metrics['moving_avg_reward'].append(moving_avg_reward)
            metrics['moving_avg_accuracy'].append(moving_avg_accuracy)
        else:
            moving_avg_reward = sum(reward_window) / len(reward_window)
            moving_avg_accuracy = sum(accuracy_window) / len(accuracy_window)
        
        metrics['time_elapsed'].append(time_elapsed)
        
        # Append losses if available
        if 'losses' in agent_metrics and agent_metrics['losses']:
            metrics['loss'].extend(agent_metrics['losses'])
        
        # Update metrics CSV
        with open(metrics_path, 'a') as f:
            f.write(f"{episode},{episode_reward},{accuracy},{epsilon},{time_elapsed},{moving_avg_reward},{moving_avg_accuracy}\n")
        
        # Log progress
        episode_time = time.time() - episode_start_time
        print(f"Episode {episode}/{NUM_EPISODES} - Reward: {episode_reward:.2f}, Accuracy: {accuracy:.2f}%, " 
              f"Epsilon: {epsilon:.4f}, Time: {episode_time:.2f}s")
        
        if len(reward_window) >= MOVING_AVG_WINDOW:
            print(f"Moving Avg (last {MOVING_AVG_WINDOW} episodes) - Reward: {moving_avg_reward:.2f}, Accuracy: {moving_avg_accuracy:.2f}%")
        
        # Periodic evaluation and checkpointing
        if episode % EVAL_FREQ == 0:
            # Log to training log
            with open(log_path, 'a') as f:
                f.write(f"Episode {episode}: Reward={episode_reward:.2f}, Accuracy={accuracy:.2f}%, ")
                f.write(f"Avg Reward={moving_avg_reward:.2f}, Avg Accuracy={moving_avg_accuracy:.2f}%, ")
                f.write(f"Epsilon={epsilon:.4f}, Time={time_elapsed:.1f}s\n")
            
            # Update visualizations
            plot_training_metrics(metrics, OUTPUT_DIR)
        
        # Save checkpoint
        if episode % CHECKPOINT_FREQ == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f'model_episode_{episode}.pth')
            agent.save(checkpoint_path)
            print(f"Saved checkpoint at episode {episode} to {checkpoint_path}")
        
        # Early stopping logic
        if len(reward_window) == MOVING_AVG_WINDOW:
            if moving_avg_reward > best_avg_reward:
                best_avg_reward = moving_avg_reward
                # Save best model
                best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
                agent.save(best_model_path)
                print(f"New best model with avg reward {best_avg_reward:.2f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {episode} episodes")
                break
    
    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"Episodes completed: {episode}/{NUM_EPISODES}")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    if len(metrics['moving_avg_reward']) > 0:
        print(f"Final moving avg reward: {metrics['moving_avg_reward'][-1]:.2f}")
        print(f"Final moving avg accuracy: {metrics['moving_avg_accuracy'][-1]:.2f}%")
    
    print(f"Best moving avg reward: {best_avg_reward:.2f}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # Final model save
    final_model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
    agent.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Final metrics update
    plot_training_metrics(metrics, OUTPUT_DIR)
    
    # Save training summary
    summary_path = os.path.join(OUTPUT_DIR, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Model 1 Training Summary\n")
        f.write("=====================\n\n")
        f.write(f"Training completed on: {datetime.now()}\n")
        f.write(f"Total episodes: {episode}\n")
        f.write(f"Training duration: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n\n")
        f.write(f"Best moving avg reward: {best_avg_reward:.2f}\n")
        if len(metrics['moving_avg_reward']) > 0:
            f.write(f"Final moving avg reward: {metrics['moving_avg_reward'][-1]:.2f}\n")
            f.write(f"Final moving avg accuracy: {metrics['moving_avg_accuracy'][-1]:.2f}%\n")
        f.write(f"Final epsilon: {metrics['epsilon'][-1]:.4f}\n\n")
        f.write("Model paths:\n")
        f.write(f"- Best model: {best_model_path}\n")
        f.write(f"- Final model: {final_model_path}\n")
    
    return True

# ===== MAIN ENTRY POINT =====
if __name__ == "__main__":
    print(f"Starting Model 1 training with {NUM_EPISODES} episodes")
    success = train_model1()
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed!")