import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
    
    # Plot winner probability distribution
    plt.figure(figsize=(10, 6))
    
    # Test set winner probabilities
    actual_winners_test = test_predictions[test_predictions['is_winner'] == 1]['win_probability']
    plt.hist(actual_winners_test, bins=20, alpha=0.5, label='Test Winners')
    
    # Holdout set winner probabilities
    actual_winners_holdout = holdout_predictions[holdout_predictions['is_winner'] == 1]['win_probability']
    plt.hist(actual_winners_holdout, bins=20, alpha=0.5, label='Holdout Winners')
    
    plt.title('Distribution of Predicted Win Probabilities for Actual Winners')
    plt.xlabel('Predicted Win Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'winner_probability_distribution.png'))
    
    # Additional visualization: Top-N accuracy comparison
    n_values = [1, 3, 5, 10]
    test_acc = [test_metrics['accuracy'], test_metrics['top3_accuracy'], 
                test_metrics['top5_accuracy'], test_metrics['top10_accuracy']]
    holdout_acc = [holdout_metrics['accuracy'], holdout_metrics['top3_accuracy'], 
                   holdout_metrics['top5_accuracy'], holdout_metrics['top10_accuracy']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, test_acc, 'o-', label='Test Set')
    plt.plot(n_values, holdout_acc, 's-', label='Holdout Set')
    plt.title('Top-N Accuracy Comparison')
    plt.xlabel('N (Top-N Prediction)')
    plt.ylabel('Accuracy')
    plt.xticks(n_values)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'top_n_accuracy.png'))
    
    # Visualize winner rank distribution
    plt.figure(figsize=(10, 6))
    
    # Get winner ranks for test and holdout sets
    winner_ranks_test = test_predictions[test_predictions['is_winner'] == 1]['predicted_rank']
    winner_ranks_holdout = holdout_predictions[holdout_predictions['is_winner'] == 1]['predicted_rank']
    
    max_rank = max(winner_ranks_test.max() if not winner_ranks_test.empty else 0,
                  winner_ranks_holdout.max() if not winner_ranks_holdout.empty else 0)
    bins = min(20, int(max_rank))
    
    plt.hist(winner_ranks_test, bins=bins, alpha=0.5, label='Test Winners')
    plt.hist(winner_ranks_holdout, bins=bins, alpha=0.5, label='Holdout Winners')
    
    plt.title('Distribution of Predicted Ranks for Actual Winners')
    plt.xlabel('Predicted Rank')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'winner_rank_distribution.png'))
    
    # Print completion message
    print("\n" + "="*50)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"All outputs have been saved to: {output_dir}")

if __name__ == "__main__":
    main()