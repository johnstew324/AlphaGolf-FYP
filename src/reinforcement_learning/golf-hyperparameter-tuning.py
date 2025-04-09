import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import pickle
import random
from collections import deque
import itertools
import multiprocessing
from functools import partial
import copy

# Import our DQN implementation
from golf_tournament_dqn import DQNAgent, ReplayBuffer

class HyperparameterTuner:
    def __init__(self, train_df, test_df, feature_list, output_dir='./tuning_results'):
    
        self.train_df = train_df
        self.test_df = test_df
        self.feature_list = feature_list
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Default parameter grid
        self.param_grid = {
            'learning_rate': [0.01, 0.005, 0.001, 0.0005],
            'batch_size': [16, 32, 64],
            'hidden_layers': [[128, 64], [128, 64, 32], [64, 32]],
            'gamma': [0.9, 0.95, 0.99],
            'epsilon_decay': [0.99, 0.995, 0.9975],
            'dropout_rate': [0.1, 0.2, 0.3]
        }
        
    def set_param_grid(self, param_grid):
        self.param_grid = param_grid
    
    def create_sample_dataset(self, n_tournaments=20, random_state=42):
        np.random.seed(random_state)

        all_tournaments = self.train_df['tournament_id_code'].unique()
        
        sample_tournaments = np.random.choice(
            all_tournaments, 
            size=min(n_tournaments, len(all_tournaments)), 
            replace=False
        )
        
        sample_df = self.train_df[self.train_df['tournament_id_code'].isin(sample_tournaments)].copy()
        
        print(f"Created sample dataset with {len(sample_tournaments)} tournaments and {len(sample_df)} rows")
        
        return sample_df
    
    def evaluate_parameters(self, params, sample_df, n_episodes=5, n_tournaments=10):
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        hidden_layers = params['hidden_layers']
        gamma = params['gamma']
        epsilon_decay = params['epsilon_decay']
        dropout_rate = params.get('dropout_rate', 0.2)
        
        # Initialize agent
        state_size = len(self.feature_list)
        agent = DQNAgent(
            state_size=state_size,
            action_size=2,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=epsilon_decay,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            memory_size=2000,  # Small memory size for quick evaluation
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )
        
        # Quick training
        total_reward = 0
        total_accuracy = 0
        total_loss = 0
        
        # Get unique tournaments
        tournaments = sample_df['tournament_id_code'].unique()
        
        # Training loop
        for episode in range(n_episodes):
            # Shuffle and limit tournaments
            np.random.shuffle(tournaments)
            episode_tournaments = tournaments[:n_tournaments]
            
            episode_reward = 0
            episode_correct = 0
            episode_tournaments_processed = 0
            episode_loss = 0
            batch_count = 0
            
            # Process each tournament
            for tournament_id in episode_tournaments:
                episode_tournaments_processed += 1
                
                # Get tournament data
                tournament_data = sample_df[sample_df['tournament_id_code'] == tournament_id].copy()
                tournament_data = tournament_data.sample(frac=1).reset_index(drop=True)  # Shuffle
                
                player_selected = False
                
                # Process each player
                for i in range(len(tournament_data)):
                    player = tournament_data.iloc[i]
                    state = player[self.feature_list].values.reshape(1, -1)
                    
                    # Choose action
                    action = agent.act(state)
                    
                    # If selected as winner
                    if action == 1:
                        player_selected = True
                        
                        # Check if correct
                        if player['is_winner'] == 1:
                            reward = 10
                            episode_correct += 1
                            episode_reward += reward
                        else:
                            reward = -5
                            episode_reward += reward
                        
                        # Store experience
                        agent.remember(state, action, reward, state, True)
                        break
                    else:  # Skip player
                        if player['is_winner'] == 1:
                            reward = -1
                            episode_reward += reward
                        
                        # Check if last player
                        if i == len(tournament_data) - 1:
                            # Force selection of last player
                            reward = -2  # Penalty for indecision
                            agent.remember(state, 1, reward, state, True)
                        else:
                            # Regular skip
                            next_player = tournament_data.iloc[i+1]
                            next_state = next_player[self.feature_list].values.reshape(1, -1)
                            agent.remember(state, action, 0, next_state, False)
                
                # Train after each tournament if enough samples
                if agent.memory.size() >= agent.batch_size:
                    loss = agent.replay()
                    episode_loss += loss
                    batch_count += 1
            
            # Calculate episode metrics
            if episode_tournaments_processed > 0:
                episode_reward /= episode_tournaments_processed
                episode_accuracy = episode_correct / episode_tournaments_processed
            else:
                episode_reward = 0
                episode_accuracy = 0
                
            if batch_count > 0:
                episode_loss /= batch_count
            else:
                episode_loss = 0

            total_reward += episode_reward
            total_accuracy += episode_accuracy
            total_loss += episode_loss
        avg_reward = total_reward / n_episodes if n_episodes > 0 else 0
        avg_accuracy = total_accuracy / n_episodes if n_episodes > 0 else 0
        avg_loss = total_loss / n_episodes if n_episodes > 0 else 0
        
        # Add final epsilon value
        final_epsilon = agent.epsilon
        
        # Create result dictionary
        result = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'hidden_layers': str(hidden_layers),  # Convert to string for DataFrame storage
            'gamma': gamma,
            'epsilon_decay': epsilon_decay,
            'dropout_rate': dropout_rate,
            'avg_reward': avg_reward,
            'avg_accuracy': avg_accuracy,
            'avg_loss': avg_loss,
            'final_epsilon': final_epsilon
        }
        
        return result
    
    def grid_search(self, sample_size=20, n_episodes=5, n_tournaments=10, n_jobs=None):
        print(f"Starting grid search with {sample_size} tournaments, {n_episodes} episodes per evaluation")
        
        # Create sample dataset
        sample_df = self.create_sample_dataset(n_tournaments=sample_size)
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            self.param_grid['learning_rate'],
            self.param_grid['batch_size'],
            self.param_grid['hidden_layers'],
            self.param_grid['gamma'],
            self.param_grid['epsilon_decay'],
            self.param_grid['dropout_rate']
        ))
        
        print(f"Total parameter combinations: {len(param_combinations)}")
        param_dicts = []
        for combo in param_combinations:
            param_dicts.append({
                'learning_rate': combo[0],
                'batch_size': combo[1],
                'hidden_layers': combo[2],
                'gamma': combo[3],
                'epsilon_decay': combo[4],
                'dropout_rate': combo[5]
            })
        evaluate_func = partial(
            self.evaluate_parameters,
            sample_df=sample_df,
            n_episodes=n_episodes,
            n_tournaments=n_tournaments
        )

        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        
        print(f"Running grid search with {n_jobs} parallel processes")
        start_time = datetime.now()
        
        # Run parallel evaluation
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = pool.map(evaluate_func, param_dicts)
        
        # Calculate total time
        total_time = datetime.now() - start_time
        print(f"Grid search completed in {total_time}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by average reward (primary) and accuracy (secondary)
        results_df = results_df.sort_values(['avg_reward', 'avg_accuracy'], ascending=[False, False])
        
        # Save results
        results_path = os.path.join(self.output_dir, 'hyperparameter_tuning_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Tuning results saved to {results_path}")
        
        # Get best parameters
        best_params = results_df.iloc[0].to_dict()
        
        # Convert string representation of hidden layers back to list
        best_params['hidden_layers'] = eval(best_params['hidden_layers'])
        
        # Remove metric columns
        metric_cols = ['avg_reward', 'avg_accuracy', 'avg_loss', 'final_epsilon']
        for col in metric_cols:
            best_params.pop(col, None)
        
        # Save best parameters
        best_params_path = os.path.join(self.output_dir, 'best_hyperparameters.pkl')
        with open(best_params_path, 'wb') as f:
            pickle.dump(best_params, f)
        
        print("\nBest hyperparameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        
        # Visualize results
        self._plot_tuning_results(results_df)
        
        return results_df, best_params
    
    def _plot_tuning_results(self, results_df):
        plots_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='learning_rate', y='avg_reward', data=results_df)
        plt.title('Learning Rate vs. Average Reward')
        plt.xlabel('Learning Rate')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'learning_rate_vs_reward.png'))

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='batch_size', y='avg_reward', data=results_df)
        plt.title('Batch Size vs. Average Reward')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'batch_size_vs_reward.png'))

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='gamma', y='avg_reward', data=results_df)
        plt.title('Gamma (Discount Factor) vs. Average Reward')
        plt.xlabel('Gamma')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'gamma_vs_reward.png'))

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='epsilon_decay', y='avg_reward', data=results_df)
        plt.title('Epsilon Decay vs. Average Reward')
        plt.xlabel('Epsilon Decay')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'epsilon_decay_vs_reward.png'))

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='dropout_rate', y='avg_reward', data=results_df)
        plt.title('Dropout Rate vs. Average Reward')
        plt.xlabel('Dropout Rate')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'dropout_rate_vs_reward.png'))

        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['avg_reward'], results_df['avg_accuracy'], alpha=0.6)
        plt.title('Average Reward vs. Average Accuracy')
        plt.xlabel('Average Reward')
        plt.ylabel('Average Accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'reward_vs_accuracy.png'))
        
        plt.close('all')
        
    def random_search(self, n_iterations=20, sample_size=20, n_episodes=5, n_tournaments=10, n_jobs=None):
        print(f"Starting random search with {n_iterations} iterations")

        sample_df = self.create_sample_dataset(n_tournaments=sample_size)
        
        # Generate random parameter combinations
        param_dicts = []
        for _ in range(n_iterations):
            params = {
                'learning_rate': random.choice(self.param_grid['learning_rate']),
                'batch_size': random.choice(self.param_grid['batch_size']),
                'hidden_layers': random.choice(self.param_grid['hidden_layers']),
                'gamma': random.choice(self.param_grid['gamma']),
                'epsilon_decay': random.choice(self.param_grid['epsilon_decay']),
                'dropout_rate': random.choice(self.param_grid['dropout_rate'])
            }
            param_dicts.append(params)
        
        # Define evaluation function for parallel processing
        evaluate_func = partial(
            self.evaluate_parameters,
            sample_df=sample_df,
            n_episodes=n_episodes,
            n_tournaments=n_tournaments
        )
        
        # Determine number of processes
        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        
        print(f"Running random search with {n_jobs} parallel processes")
        start_time = datetime.now()
        
        # Run parallel evaluation
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = pool.map(evaluate_func, param_dicts)
        

        total_time = datetime.now() - start_time
        print(f"Random search completed in {total_time}")

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(['avg_reward', 'avg_accuracy'], ascending=[False, False])

        results_path = os.path.join(self.output_dir, 'random_search_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Random search results saved to {results_path}")

        best_params = results_df.iloc[0].to_dict()

        best_params['hidden_layers'] = eval(best_params['hidden_layers'])
    
        metric_cols = ['avg_reward', 'avg_accuracy', 'avg_loss', 'final_epsilon']
        for col in metric_cols:
            best_params.pop(col, None)
        

        best_params_path = os.path.join(self.output_dir, 'best_random_search_params.pkl')
        with open(best_params_path, 'wb') as f:
            pickle.dump(best_params, f)
        
        print("\nBest hyperparameters found by random search:")
        for param, value in best_params.items():
            print(f"{param}: {value}")

        self._plot_tuning_results(results_df)
        
        return results_df, best_params


if __name__ == "__main__":
    import argparse
    from golf_tournament_dqn import data_preparation
    
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Golf DQN Model')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--output', type=str, default='./tuning_results', help='Output directory')
    parser.add_argument('--method', type=str, choices=['grid', 'random'], default='random',
                        help='Tuning method: grid search or random search')
    parser.add_argument('--iterations', type=int, default=20, 
                        help='Number of iterations for random search')
    parser.add_argument('--sample_size', type=int, default=20,
                        help='Number of tournaments to use in sample dataset')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes for each evaluation')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"tuning_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output will be saved to: {output_dir}")


    print("Preparing data...")
    train_df, test_df, _, feature_list = data_preparation(args.data)
    
    tuner = HyperparameterTuner(train_df, test_df, feature_list, output_dir=output_dir)

    if args.method == 'grid':
        print("Running grid search...")
        results_df, best_params = tuner.grid_search(
            sample_size=args.sample_size,
            n_episodes=args.episodes,
            n_jobs=args.jobs
        )
    else:
        print("Running random search...")
        results_df, best_params = tuner.random_search(
            n_iterations=args.iterations,
            sample_size=args.sample_size,
            n_episodes=args.episodes,
            n_jobs=args.jobs
        )
    
    print("Hyperparameter tuning completed!")