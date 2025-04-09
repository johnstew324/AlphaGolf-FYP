import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

# Import our module
from golf_tournament_dqn import (
    data_preparation,
    DQNAgent,
    train_dqn_agent,
    evaluate_dqn_agent,
    plot_training_history
)

def main():
    """Main execution function for Golf Tournament DQN prediction system"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Golf Tournament Winner Prediction with DQN')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to the dataset CSV file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pre-trained model (optional)')
    parser.add_argument('--output', type=str, default='./outputs', 
                        help='Output directory')
    parser.add_argument('--episodes', type=int, default=100, 
                        help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--tune', action='store_true', 
                        help='Perform hyperparameter tuning')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation (requires --model)')
    
    args = parser.parse_args()
    
    # Check if we're in evaluation-only mode
    if args.eval_only and args.model is None:
        print("Error: --eval_only requires --model to be specified")
        return
    
    # Set up output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output will be saved to: {output_dir}")
    
    # Step 1: Data Preparation
    print("\n" + "="*50)
    print("STEP 1: DATA PREPARATION")
    print("="*50)
    
    train_df, test_df, holdout_df, feature_list = data_preparation(args.data)
    
    # Save the processed datasets
    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    holdout_df.to_csv(os.path.join(output_dir, 'holdout_data.csv'), index=False)
    
    # Save feature list
    with open(os.path.join(output_dir, 'feature_list.txt'), 'w') as f:
        f.write('\n'.join(feature_list))
    
    print(f"\nProcessed datasets saved to {output_dir}")
    
    # If we're just evaluating, skip to step 3
    if args.eval_only:
        print("\nSkipping training, loading pre-trained model for evaluation")
        
        # Initialize agent with correct state size
        state_size = len(feature_list)
        agent = DQNAgent(state_size=state_size)
        
        # Load pre-trained model
        agent.load_model(args.model)
        
    else:
        # Step 2: Model Training
        print("\n" + "="*50)
        print("STEP 2: MODEL TRAINING")
        print("="*50)
        
        # Train model or use pre-trained
        if args.model:
            print(f"Loading pre-trained model from {args.model}")
            state_size = len(feature_list)
            agent = DQNAgent(state_size=state_size)
            agent.load_model(args.model)
        else:
            print("Training new DQN model...")
            
            # Configure training parameters
            agent, history = train_dqn_agent(
                train_df=train_df,
                test_df=test_df,
                feature_list=feature_list,
                num_episodes=args.episodes,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                model_dir=output_dir,
                auto_tune=args.tune
            )
            
            # Save training history
            history_df = pd.DataFrame({
                'episode': range(1, args.episodes + 1),
                'reward': history['episode_rewards'],
                'accuracy': history['episode_accuracy'],
                'loss': history['episode_loss'],
                'epsilon': history['epsilon_values'],
                'time': history['time_per_episode']
            })
            
            history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Step 3: Model Evaluation
    print("\n" + "="*50)
    print("STEP 3: MODEL EVALUATION")
    print("="*50)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_metrics, test_predictions = evaluate_dqn_agent(agent, test_df, feature_list)
    
    # Save test predictions
    test_predictions.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    # Evaluate on holdout set
    print("\nEvaluating model on holdout set...")
    holdout_metrics, holdout_predictions = evaluate_dqn_agent(agent, holdout_df, feature_list)
    
    # Save holdout predictions
    holdout_predictions.to_csv(os.path.join(output_dir, 'holdout_predictions.csv'), index=False)
    
    # Save metrics summary
    metrics_summary = pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Test Set': list(test_metrics.values()),
        'Holdout Set': list(holdout_metrics.values())
    })
    
    metrics_summary.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
    
    # Print evaluation summary
    print("\nEvaluation Summary:")
    print(metrics_summary)
    
     # Step 4: Generate visualizations
    print("\n" + "="*50)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*50)
     # Plot winner-specific visualizations
    plot_winner_prediction_results(test_predictions, output_dir)
    plot_winner_prediction_results(holdout_predictions, output_dir) 
    
    # Create extended metrics visualization for all targets
    target_names = ['hist_winner', 'hist_top3', 'hist_top10', 'hist_top25', 'hist_made_cut']
    available_targets = [t for t in target_names if f'{t}_precision' in holdout_metrics]
    
    if available_targets:
        # Create a new visualization for target metrics
        plt.figure(figsize=(12, 8))
        
        # Plot F1 scores for each target
        metrics_to_plot = []
        for target in available_targets:
            metrics_to_plot.append({
                'Target': target.replace('hist_', ''),
                'Set': 'Test',
                'Precision': test_metrics[f'{target}_precision'],
                'Recall': test_metrics[f'{target}_recall'],
                'F1': test_metrics[f'{target}_f1']
            })
            metrics_to_plot.append({
                'Target': target.replace('hist_', ''),
                'Set': 'Holdout',
                'Precision': holdout_metrics[f'{target}_precision'],
                'Recall': holdout_metrics[f'{target}_recall'],
                'F1': holdout_metrics[f'{target}_f1']
            })
        
        metrics_df = pd.DataFrame(metrics_to_plot)
        
        # Reshaping data for seaborn
        melted_df = pd.melt(metrics_df, 
                           id_vars=['Target', 'Set'], 
                           value_vars=['Precision', 'Recall', 'F1'],
                           var_name='Metric', value_name='Value')
        
        # Create grouped bar plot
        sns.barplot(x='Target', y='Value', hue='Set', col='Metric', data=melted_df)
        plt.title('Prediction Performance Across Target Levels')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'target_metrics.png'))
        
        # Plot comparison of precision vs recall for each target
        plt.figure(figsize=(12, 8))
        plt.plot([test_metrics[f'{t}_recall'] for t in available_targets], 
                [test_metrics[f'{t}_precision'] for t in available_targets], 
                'o-', label='Test Set')
        plt.plot([holdout_metrics[f'{t}_recall'] for t in available_targets], 
                [holdout_metrics[f'{t}_precision'] for t in available_targets], 
                's-', label='Holdout Set')
        
        # Add labels for each point
        for i, target in enumerate(available_targets):
            plt.annotate(target.replace('hist_', ''), 
                        (test_metrics[f'{target}_recall'], test_metrics[f'{target}_precision']),
                        xytext=(5, 5), textcoords='offset points')
            plt.annotate(target.replace('hist_', ''), 
                        (holdout_metrics[f'{target}_recall'], holdout_metrics[f'{target}_precision']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Precision vs Recall for Different Target Levels')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    
    # Your existing visualizations...
    
    # Additional target-specific visualization:
    if available_targets and 'position_numeric' in holdout_predictions.columns:
        plt.figure(figsize=(10, 6))
        
        # Aggregate data
        position_data = []
        
        # For test set
        for pos in range(1, 11):
            avg_prob = test_predictions[test_predictions['position_numeric'] == pos]['win_probability'].mean()
            if not np.isnan(avg_prob):
                position_data.append({
                    'Position': pos,
                    'Set': 'Test',
                    'Win Probability': avg_prob
                })
        
        # For holdout set
        for pos in range(1, 11):
            avg_prob = holdout_predictions[holdout_predictions['position_numeric'] == pos]['win_probability'].mean()
            if not np.isnan(avg_prob):
                position_data.append({
                    'Position': pos,
                    'Set': 'Holdout',
                    'Win Probability': avg_prob
                })
        
        # Convert to DataFrame and plot
        position_df = pd.DataFrame(position_data)
        sns.lineplot(x='Position', y='Win Probability', hue='Set', data=position_df, marker='o')
        plt.title('Average Predicted Win Probability by Actual Position')
        plt.xlabel('Actual Position')
        plt.ylabel('Average Predicted Win Probability')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'position_vs_probability.png'))
        
        
        
        
        
        
        
def plot_winner_prediction_results(predictions_df, output_dir):
    """
    Create visualizations focused on winner prediction success
    
    Parameters:
    -----------
    predictions_df : DataFrame
        Prediction results with is_winner and predicted_rank columns
    output_dir : str
        Directory to save plots
    """
    # Create winner-specific visualizations
    plt.figure(figsize=(12, 8))
    
    # Distribution of predicted ranks for actual winners
    winner_ranks = predictions_df[predictions_df['is_winner'] == 1]['predicted_rank']
    
    plt.subplot(2, 2, 1)
    sns.histplot(winner_ranks, bins=20, kde=True)
    plt.title('Distribution of Predicted Ranks for Actual Winners')
    plt.xlabel('Predicted Rank')
    plt.ylabel('Count')
    plt.axvline(x=1, color='r', linestyle='--', label='Top Prediction')
    plt.axvline(x=3, color='g', linestyle='--', label='Top 3')
    plt.axvline(x=5, color='orange', linestyle='--', label='Top 5')
    plt.legend()
    
    # Distribution of predicted probabilities for winners vs non-winners
    plt.subplot(2, 2, 2)
    sns.histplot(predictions_df[predictions_df['is_winner'] == 1]['win_probability'], 
                color='green', alpha=0.5, label='Winners', kde=True)
    sns.histplot(predictions_df[predictions_df['is_winner'] == 0]['win_probability'], 
                color='red', alpha=0.5, label='Non-Winners', kde=True)
    plt.title('Distribution of Predicted Win Probabilities')
    plt.xlabel('Predicted Win Probability')
    plt.ylabel('Count')
    plt.legend()
    
    # Top-N accuracy visualization
    plt.subplot(2, 2, 3)
    n_values = list(range(1, 21))
    accuracies = []
    
    for n in n_values:
        # Calculate percentage of winners in top-n predictions
        top_n_count = len(winner_ranks[winner_ranks <= n])
        acc = top_n_count / len(winner_ranks) if len(winner_ranks) > 0 else 0
        accuracies.append(acc)
    
    plt.plot(n_values, accuracies, 'o-')
    plt.title('Top-N Accuracy for Winner Prediction')
    plt.xlabel('N (Top-N Prediction)')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # ROC-like curve for winner prediction
    plt.subplot(2, 2, 4)
    
    # Sort players by win probability and calculate cumulative number of real winners
    sorted_idx = predictions_df['win_probability'].sort_values(ascending=False).index
    sorted_df = predictions_df.loc[sorted_idx].reset_index(drop=True)
    
    total_winners = sorted_df['is_winner'].sum()
    total_players = len(sorted_df)
    
    cumulative_winners = np.cumsum(sorted_df['is_winner']) / total_winners if total_winners > 0 else 0
    fraction_players = np.arange(1, total_players + 1) / total_players
    
    plt.plot(fraction_players, cumulative_winners)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('Winner Concentration Curve')
    plt.xlabel('Fraction of Players (Sorted by Predicted Probability)')
    plt.ylabel('Fraction of Actual Winners')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'winner_prediction_analysis.png'))
    plt.close()
    
    print(f"Winner prediction analysis plot saved to {output_dir}")

if __name__ == "__main__":
    main()