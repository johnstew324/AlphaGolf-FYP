# simple_dataset_builder.py
import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from collections import defaultdict

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import components
from database import DatabaseManager
from config import Config
from data_Excator.data_excractor import DataExtractor
from feature_engineering.pipeline import FeaturePipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_builder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_tournament_targets(tournament_id, extractor):
    """
    Create target variables for a tournament using scorecard data.
    
    Args:
        tournament_id: Tournament ID in standard RYYYYTTT format
        extractor: DataExtractor instance
        
    Returns:
        DataFrame with target variables or None if no data
    """
    logger.info(f"Creating target variables for tournament {tournament_id}")
    
    # Get scorecard data
    scorecard_data = extractor.extract_player_scorecards(tournament_ids=tournament_id)
    
    if scorecard_data.empty:
        logger.warning(f"No scorecard data found for tournament {tournament_id}")
        return None
        
    logger.info(f"Found {len(scorecard_data)} scorecard records for {tournament_id}")
    
    # Group by player to extract final results
    player_results = {}
    
    for player_id in scorecard_data['player_id'].unique():
        player_cards = scorecard_data[scorecard_data['player_id'] == player_id]
        
        if player_cards.empty:
            continue
            
        # Check how many rounds the player completed
        rounds_completed = len(player_cards)
        
        # Get final round
        if 'current_round' in player_cards.columns:
            final_round_num = player_cards['current_round'].max()
        else:
            final_round_num = player_cards['round_number'].max() if 'round_number' in player_cards.columns else 4
            
        # Calculate final score
        total_score = None
        score_to_par = None
        
        # Most recent record should have their score
        latest_record = player_cards.sort_values('collected_at', ascending=False).iloc[0]
        
        # Get score if available
        if 'score_to_par' in latest_record:
            score_to_par = latest_record['score_to_par']
        
        # Store player result
        player_results[player_id] = {
            'player_id': player_id,
            'rounds_completed': rounds_completed,
            'final_round': final_round_num,
            'score_to_par': score_to_par,
            'raw_data': latest_record.to_dict()
        }
    
    if not player_results:
        logger.warning("No valid player results found")
        return None
        
    # Calculate positions from scores
    sorted_players = []
    
    # First gather players with score data
    players_with_scores = [
        (pid, data) 
        for pid, data in player_results.items() 
        if data['score_to_par'] is not None
    ]
    
    # Sort by score to par (lower is better) and rounds completed (higher is better)
    sorted_players = sorted(
        players_with_scores,
        key=lambda x: (x[1]['score_to_par'], -x[1]['rounds_completed'])
    )
    
    # Assign positions
    position = 1
    prev_score = None
    prev_rounds = None
    
    for i, (player_id, data) in enumerate(sorted_players):
        # Check for ties
        if i > 0 and data['score_to_par'] == prev_score and data['rounds_completed'] == prev_rounds:
            player_results[player_id]['position'] = position
        else:
            position = i + 1
            player_results[player_id]['position'] = position
            
        prev_score = data['score_to_par']
        prev_rounds = data['rounds_completed']
    
    # Create target variables dataframe
    target_data = []
    
    for player_id, data in player_results.items():
        # Only include players with position data
        if 'position' in data:
            position = data['position']
            
            target_row = {
                'player_id': player_id,
                'tournament_id': tournament_id,
                'position': position,
                'winner': 1 if position == 1 else 0,
                'top3': 1 if position <= 3 else 0,
                'top10': 1 if position <= 10 else 0,
                'made_cut': 1 if data['rounds_completed'] >= 3 else 0  # Assuming 4 round tournaments
            }
            
            target_data.append(target_row)
    
    if not target_data:
        logger.warning("No target data created")
        return None
        
    target_df = pd.DataFrame(target_data)
    logger.info(f"Created {len(target_df)} target records for tournament {tournament_id}")
    
    return target_df

def build_tournament_dataset(tournament_id, extractor, pipeline):
    """
    Build a complete dataset for a single tournament.
    
    Args:
        tournament_id: Tournament ID in standard RYYYYTTT format
        extractor: DataExtractor instance
        pipeline: FeaturePipeline instance
        
    Returns:
        Tuple of (features_df, targets_df) or (None, None) if error
    """
    logger.info(f"Building dataset for tournament {tournament_id}")
    
    try:
        # Extract season from tournament ID
        season = int(tournament_id[1:5])
        logger.info(f"Extracted season {season} from tournament ID")
        
        # Generate features
        logger.info(f"Generating features...")
        features = pipeline.generate_features(tournament_id, season)
        
        if features is None or features.empty:
            logger.warning(f"No features generated for {tournament_id}")
            return None, None
            
        logger.info(f"Generated features: {features.shape}")
        
        # Create target variables
        logger.info(f"Creating target variables...")
        targets = create_tournament_targets(tournament_id, extractor)
        
        if targets is None or targets.empty:
            logger.warning(f"No target variables created for {tournament_id}")
            return features, None
            
        logger.info(f"Created target variables: {targets.shape}")
        
        return features, targets
        
    except Exception as e:
        logger.error(f"Error building dataset for tournament {tournament_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def build_master_dataset(tournament_ids, output_dir='data', feature_sets_path=None):
    """
    Build master dataset from multiple tournaments.
    
    Args:
        tournament_ids: List of tournament IDs
        output_dir: Directory to save dataset files
        feature_sets_path: Path to optimized feature sets JSON
        
    Returns:
        Path to master dataset or None if failed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load optimized feature sets if provided
    optimized_sets = None
    if feature_sets_path and os.path.exists(feature_sets_path):
        try:
            with open(feature_sets_path, 'r') as f:
                feature_data = json.load(f)
                if 'optimized_sets' in feature_data:
                    optimized_sets = feature_data['optimized_sets']
                    logger.info(f"Loaded optimized feature sets for {len(optimized_sets)} targets")
        except Exception as e:
            logger.error(f"Failed to load feature sets: {str(e)}")
    
    # Initialize database and extractor
    logger.info("Connecting to database...")
    db_manager = DatabaseManager(uri=Config.MONGODB_URI, database_name="pga_tour_data")
    extractor = DataExtractor(db_manager)
    
    # Create pipeline
    pipeline = FeaturePipeline(extractor)
    
    # Process each tournament
    all_features = []
    all_targets = []
    successful_tournaments = []
    
    for tournament_id in tournament_ids:
        logger.info(f"\n--- Processing tournament {tournament_id} ---")
        
        features, targets = build_tournament_dataset(tournament_id, extractor, pipeline)
        
        if features is not None and not features.empty:
            # Ensure tournament_id is present
            if 'tournament_id' not in features.columns:
                features['tournament_id'] = tournament_id
                
            all_features.append(features)
            logger.info(f"Added features for tournament {tournament_id}")
            
            if targets is not None and not targets.empty:
                all_targets.append(targets)
                successful_tournaments.append(tournament_id)
                logger.info(f"Added targets for tournament {tournament_id}")
            else:
                logger.warning(f"Missing targets for tournament {tournament_id}")
        else:
            logger.warning(f"Skipping tournament {tournament_id} - no features generated")
    
    # Check if we have any successful data
    if not all_features or not all_targets:
        logger.error("No valid data collected")
        return None
        
    # Combine all data
    logger.info("\nCombining all tournament data...")
    features_df = pd.concat(all_features, ignore_index=True)
    targets_df = pd.concat(all_targets, ignore_index=True)
    
    logger.info(f"Combined features shape: {features_df.shape}")
    logger.info(f"Combined targets shape: {targets_df.shape}")
    
    
    conflicting_cols = ['top10', 'top3', 'winner', 'made_cut']  # List potential conflicts
    for col in conflicting_cols:
        if col in features_df.columns and col in targets_df.columns:
            # Rename the column in the features dataframe
            features_df = features_df.rename(columns={col: f"{col}_feat"})
            logger.info(f"Renamed column '{col}' to '{col}_feat' in features to avoid conflict")
    
    # Merge features and targets
    logger.info("Creating master dataset...")
    master_df = pd.merge(
        features_df, 
        targets_df,
        on=['player_id', 'tournament_id'],
        how='inner'
    )
    
    logger.info(f"Master dataset shape: {master_df.shape}")
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_path = os.path.join(output_dir, f'master_dataset_{timestamp}.csv')
    
    logger.info(f"Saving master dataset to {master_path}")
    master_df.to_csv(master_path, index=False)
    
    # Create train/test split info
    if successful_tournaments:
        # Create chronological split
        train_size = int(len(successful_tournaments) * 0.8)
        
        train_tournaments = successful_tournaments[:train_size]
        test_tournaments = successful_tournaments[train_size:]
        
        split_info = {
            'train_tournaments': train_tournaments,
            'test_tournaments': test_tournaments,
            'timestamp': timestamp
        }
        
        # Save split info
        split_path = os.path.join(output_dir, f'train_test_split_{timestamp}.json')
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
            
        # Create train/test datasets
        train_df = master_df[master_df['tournament_id'].isin(train_tournaments)]
        test_df = master_df[master_df['tournament_id'].isin(test_tournaments)]
        
        train_path = os.path.join(output_dir, f'train_dataset_{timestamp}.csv')
        test_path = os.path.join(output_dir, f'test_dataset_{timestamp}.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Created train dataset with {len(train_df)} rows")
        logger.info(f"Created test dataset with {len(test_df)} rows")
    
    # Create target-specific datasets if optimized sets are available
    if optimized_sets:
        target_paths = {}
        
        for target_type, feature_list in optimized_sets.items():
            # Get target column
            target_map = {
                'win': 'winner',
                'cut': 'made_cut',
                'top3': 'top3', 
                'top10': 'top10'
            }
            
            target_col = target_map.get(target_type, target_type)
            
            if target_col not in master_df.columns:
                logger.warning(f"Target column '{target_col}' not in master dataset")
                continue
                
            # Filter for features that exist
            valid_features = [f for f in feature_list if f in master_df.columns]
            
            # Add essential columns
            for col in ['player_id', 'tournament_id', target_col]:
                if col not in valid_features and col in master_df.columns:
                    valid_features.append(col)
            
            # Create target dataset
            target_df = master_df[valid_features]
            target_path = os.path.join(output_dir, f'{target_type}_dataset_{timestamp}.csv')
            
            target_df.to_csv(target_path, index=False)
            logger.info(f"Created {target_type} dataset with {len(valid_features)} features")
            
            target_paths[target_type] = target_path
    
    return master_path

if __name__ == "__main__":
    # Define tournaments to include
    tournament_ids = [
        "R2025016",  # The Sentry 2025
        "R2024016",  # The Sentry 2024
        "R2024003",  # Tournament 2 from 2025s
        "R2023007"   # Tournament 4 from 2025
    ]
    
    # Path to optimized feature sets
    feature_sets_path = 'feature_refinement/optimized_feature_sets.json'
    
    # Build the dataset
    logger.info("Starting dataset build process...")
    master_path = build_master_dataset(tournament_ids, feature_sets_path=feature_sets_path)
    
    if master_path:
        logger.info(f"Success! Master dataset created at: {master_path}")
    else:
        logger.error("Dataset build failed")