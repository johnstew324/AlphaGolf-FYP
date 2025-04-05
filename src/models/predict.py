"""
Prediction script for DQN models.
This script uses trained DQN models to predict golf tournament outcomes.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import json
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from models.dqn_model import DQNAgent, QNetwork
from models.golf_tournament_env import GolfTournamentEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("models/prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict golf tournament outcomes using trained DQN models")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="processed_data/enhanced_dqn_data.csv", 
                        help="Path to the dataset file")
    parser.add_argument("--target", type=str, default="winner", 
                        choices=["winner", "top10_finish", "made_cut"], 
                        help="Prediction target")
    parser.add_argument("--model_path", type=str, default="models/trained/best_dqn_winner.pth", 
                        help="Path to the trained model file")
    parser.add_argument("--output_dir", type=str, default="models/predictions", 
                        help="Directory to save prediction results")
    parser.add_argument("--tournament_id", type=str, default=None, 
                        help="Specific tournament ID to predict (if None, predict all tournaments)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Decision threshold for predictions (0.0-1.0)")
    parser.add_argument("--top_k", type=int, default=10, 
                        help="Number of top predictions to include")
    
    # Device arguments
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cpu, cuda, or None for auto)")
    
    return parser.parse_args()

def load_model(args: argparse.Namespace) -> DQNAgent:
    """
    Load a trained DQN model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Loaded DQN agent
    """
    # Create environment to get state and action dimensions
    env = GolfTournamentEnv(
        data_path=args.data_path,
        prediction_target=args.target
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=args.device
    )
    
    # Load model
    agent.load(args.model_path)
    logger.info(f"Loaded model from {args.model_path}")
    
    return agent

def predict_tournament(agent: DQNAgent, 
                      data: pd.DataFrame, 
                      tournament_id: str,
                      target: str,
                      feature_columns: List[str],
                      threshold: float = 0.5,
                      top_k: int = 10) -> Dict[str, Any]:
    """
    Make predictions for a specific tournament.
    
    Args:
        agent: Trained DQN agent
        data: Dataset DataFrame
        tournament_id: Tournament ID to predict
        target: Prediction target
        feature_columns: Feature columns to use
        threshold: Decision threshold for predictions
        top_k: Number of top predictions to include
        
    Returns:
        Dictionary with prediction results
    """
    # Filter data for tournament
    tournament_data = data[data['tournament_id'] == tournament_id].copy()
    
    if tournament_data.empty:
        logger.warning(f"No data found for tournament {tournament_id}")
        return None
    
    # Extract features
    X = tournament_data[feature_columns].values.astype(np.float32)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X).to(agent.device)
    
    # Get player IDs and names
    player_ids = tournament_data['player_id'].values
    player_names = tournament_data['player_name'].values
    
    # Get Q-values from policy network
    with torch.no_grad():
        agent.policy_network.eval()
        q_values = agent.policy_network(X_tensor)
    
    # Convert to numpy
    q_values_np = q_values.cpu().numpy()
    
    # Get positive class probability (action 1)
    positive_probs = q_values_np[:, 1] / np.sum(q_values_np, axis=1)
    
    # Make binary predictions using threshold
    predictions = (positive_probs >= threshold).astype(int)
    
    # Get actual values if available
    has_actual = target in tournament_data.columns
    actual = tournament_data[target].values if has_actual else None
    
    # Calculate metrics if actual values are available
    metrics = {}
    if has_actual:
        metrics['accuracy'] = np.mean(predictions == actual)
        true_positives = np.sum((predictions == 1) & (actual == 1))
        false_positives = np.sum((predictions == 1) & (actual == 0))
        false_negatives = np.sum((predictions == 0) & (actual == 1))
        true_negatives = np.sum((predictions == 0) & (actual == 0))
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        metrics['true_positives'] = int(true_positives)
        metrics['false_positives'] = int(false_positives)
        metrics['false_negatives'] = int(false_negatives)
        metrics['true_negatives'] = int(true_negatives)
    
    # Create result dictionary
    result = {
        'tournament_id': tournament_id,
        'num_players': len(tournament_data),
        'predictions': []
    }
    
    # Add metrics if available
    if metrics:
        result['metrics'] = metrics
    
    # Create player predictions
    player_predictions = []
    for i in range(len(tournament_data)):
        player_dict = {
            'player_id': str(player_ids[i]),
            'player_name': player_names[i],
            'probability': float(positive_probs[i]),
            'prediction': int(predictions[i])
        }
        
        # Add actual value if available
        if has_actual:
            player_dict['actual'] = int(actual[i])
            player_dict['correct'] = int(predictions[i] == actual[i])
        
        player_predictions.append(player_dict)
    
    # Sort by probability (descending)
    player_predictions = sorted(player_predictions, key=lambda x: x['probability'], reverse=True)
    
    # Add all predictions to result
    result['predictions'] = player_predictions
    
    # Add top_k predictions
    result['top_predictions'] = player_predictions[:top_k]
    
    # Return result
    return result

def predict_tournaments(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Make predictions for tournaments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with prediction results
    """
    # Load data
    data = pd.read_csv(args.data_path)
    
    # Load model
    agent = load_model(args)
    
    # Get feature columns
    exclude_columns = [
        'player_id', 'player_name', 'tournament_id', 'year',
        'reward', 'normalized_reward', 'position', 'winner',
        'made_cut', 'top10_finish', 'top20_finish'
    ]
    
    # Get all numeric columns except excluded ones
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    # Get unique tournaments
    if args.tournament_id:
        # Specific tournament
        tournament_ids = [args.tournament_id]
    else:
        # All tournaments
        tournament_ids = data['tournament_id'].unique()
    
    # Make predictions for each tournament
    all_results = []
    overall_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'tournaments': 0
    }
    
    for tournament_id in tournament_ids:
        logger.info(f"Predicting tournament {tournament_id}")
        
        # Make predictions
        result = predict_tournament(
            agent=agent,
            data=data,
            tournament_id=tournament_id,
            target=args.target,
            feature_columns=feature_columns,
            threshold=args.threshold,
            top_k=args.top_k
        )
        
        if result:
            all_results.append(result)
            
            # Update overall metrics
            if 'metrics' in result:
                overall_metrics['accuracy'] += result['metrics']['accuracy']
                overall_metrics['precision'] += result['metrics']['precision']
                overall_metrics['recall'] += result['metrics']['recall']
                overall_metrics['f1_score'] += result['metrics']['f1_score']
                overall_metrics['tournaments'] += 1
    
    # Calculate averages for overall metrics
    if overall_metrics['tournaments'] > 0:
        for key in ['accuracy', 'precision', 'recall', 'f1_score']:
            overall_metrics[key] /= overall_metrics['tournaments']
    
    # Create summary
    summary = {
        'target': args.target,
        'model_path': args.model_path,
        'num_tournaments': len(all_results),
        'overall_metrics': overall_metrics,
        'tournament_results': all_results
    }
    
    return summary

def odds_to_ev(probability: float, odds: float) -> float:
    """
    Calculate expected value from probability and odds.
    
    Args:
        probability: Predicted probability of outcome
        odds: Decimal odds for outcome
        
    Returns:
        Expected value
    """
    # Expected value = probability * (odds - 1) - (1 - probability) * 1
    return probability * (odds - 1) - (1 - probability)

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file {args.data_path} not found")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file {args.model_path} not found")
        return
    
    # Make predictions
    logger.info(f"Making predictions for target: {args.target}")
    results = predict_tournaments(args)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"predictions_{args.target}_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved prediction results to {output_path}")
    
    # Print overall metrics
    if 'overall_metrics' in results:
        metrics = results['overall_metrics']
        logger.info("Overall metrics:")
        logger.info(f"  Tournaments: {metrics['tournaments']}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    # Print top predictions for tournaments
    for i, result in enumerate(results.get('tournament_results', [])):
        logger.info(f"\nTournament {i+1}/{len(results.get('tournament_results', []))}: {result['tournament_id']}")
        logger.info(f"Top {args.top_k} predictions:")
        for j, player in enumerate(result.get('top_predictions', [])):
            logger.info(f"  {j+1}. {player.get('player_name', '')} - Probability: {player.get('probability', 0):.4f}")
            if 'actual' in player:
                logger.info(f"     Actual: {player['actual']}, Correct: {player['correct']}")

if __name__ == "__main__":
    from datetime import datetime
    main() 