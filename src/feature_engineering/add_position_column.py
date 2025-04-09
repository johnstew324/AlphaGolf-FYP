#!/usr/bin/env python
# add_position_column.py
import pandas as pd
import os
import sys
from typing import List, Dict, Optional, Union
import numpy as np

# Add parent directory to path to import your modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Path to your features CSV
INPUT_FEATURES_PATH = "winner_specific_features.csv"  # Update this to your actual file path
OUTPUT_PATH = "features_set_with_winners.csv"  # Output path

# Import your data extractor (adjust import path as needed)
from data_excrator.data_excractor import DataExtractor
from database import DatabaseManager
from config import Config

import logging


# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import your modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def convert_tournament_id(tournament_id):
    """Convert a tournament ID from RYYYYTTT format to R2025TTT format."""
    if tournament_id.startswith("R") and len(tournament_id) >= 8:
        special_id = "R2025" + tournament_id[5:]
        return special_id
    return tournament_id

def parse_position(position):
    """
    Parse tournament position strings to numeric values, handling special cases:
    - 'T3' -> 3.0 (tied for 3rd)
    - 'CU', 'CUT' -> 101.0 (cut)
    - 'WD', 'W/D' -> 102.0 (withdrawn)
    - 'DQ' -> 103.0 (disqualified)
    - 'P2' -> 2.0 (playoff for 2nd)
    """
    if position is None:
        return None
    
    position_str = str(position).upper()
    
    # Handle numeric positions
    if position_str.isdigit():
        return float(position_str)
    
    # Handle tied positions (e.g., T3)
    if position_str.startswith('T') and position_str[1:].isdigit():
        return float(position_str[1:])
    
    # Handle playoff positions (e.g., P2)
    if position_str.startswith('P') and position_str[1:].isdigit():
        return float(position_str[1:])
    
    # Handle special cases
    special_cases = {
        'CU': 101.0,    # Cut
        'CUT': 101.0,   # Cut (alternate format)
        'WD': 102.0,    # Withdrawn
        'W/D': 102.0,   # Withdrawn (alternate format)
        'DQ': 103.0,    # Disqualified
        'DSQ': 103.0,   # Disqualified (alternate format)
        'DNS': 104.0    # Did not start
    }
    
    if position_str in special_cases:
        return special_cases[position_str]
    
    # If we can't parse it, log and return None
    logger.warning(f"Unrecognized position format: '{position_str}'")
    return None

def extract_all_tournament_history(data_extractor, tournament_id):
    """
    Extract tournament history without player filtering first,
    then manually filter player data from the raw documents.
    Ensures only one record per player is kept.
    """
    logger.info(f"Extracting all tournament history for {tournament_id}")
    
    # Convert to history tournament ID format
    history_tournament_id = convert_tournament_id(tournament_id)
    logger.info(f"Using history ID: {history_tournament_id}")
    
    # Get raw tournament history documents
    try:
        query = {"tournament_id": history_tournament_id}
        history_docs = data_extractor.db_manager.run_query("tournament_history", query)
        
        if not history_docs:
            logger.warning(f"No tournament history found for {history_tournament_id}")
            return pd.DataFrame()
            
        logger.info(f"Found {len(history_docs)} tournament history documents")
        
        # To prevent duplicates, use a dictionary to track unique player entries
        # We'll keep the entry with the best position for each player
        player_data_dict = {}
        
        for doc in history_docs:
            # Get tournament base info
            tournament_base = {
                "tournament_id": tournament_id,  # Use original tournament ID
                "history_tournament_id": doc.get("tournament_id"),
                "year": doc.get("year")
            }
            
            # Extract player data
            players = doc.get("players", [])
            if players:
                logger.info(f"Found {len(players)} players in tournament document")
                
                for player in players:
                    if isinstance(player, dict) and "player_id" in player:
                        player_id = str(player.get("player_id"))
                        
                        # Get position information and parse it
                        raw_position = player.get("position")
                        position_numeric = parse_position(raw_position)
                        
                        if position_numeric is not None:
                            # Create player entry
                            player_entry = tournament_base.copy()
                            player_entry["player_id"] = player_id
                            player_entry["player_name"] = player.get("name")
                            player_entry["position_numeric"] = position_numeric
                            player_entry["original_position"] = raw_position
                            
                            # Add target variables
                            player_entry["hist_winner"] = 1 if position_numeric == 1 else 0
                            player_entry["hist_top3"] = 1 if position_numeric <= 3 else 0
                            player_entry["hist_top10"] = 1 if position_numeric <= 10 else 0
                            player_entry["hist_top25"] = 1 if position_numeric <= 25 else 0
                            player_entry["hist_made_cut"] = 1 if position_numeric < 100 else 0
                            
                            # Use player_id as key for the dictionary
                            player_key = f"{player_id}_{tournament_id}"
                            
                            # If player already exists, only keep the better position
                            if player_key in player_data_dict:
                                existing_position = player_data_dict[player_key]["position_numeric"]
                                # Keep the better (lower) position
                                if position_numeric < existing_position:
                                    player_data_dict[player_key] = player_entry
                            else:
                                player_data_dict[player_key] = player_entry
        
        # Convert dictionary to list for DataFrame creation
        player_data = list(player_data_dict.values())
        
        # Create DataFrame from extracted player data
        if player_data:
            player_df = pd.DataFrame(player_data)
            logger.info(f"Extracted {len(player_df)} unique player records with position data")
            
            # Count special cases
            if 'original_position' in player_df.columns:
                position_counts = player_df['original_position'].value_counts()
                logger.info(f"Position distribution: {position_counts.head(10)}")
                
                # Count special cases
                cut_count = player_df[player_df['position_numeric'] == 101.0].shape[0]
                wd_count = player_df[player_df['position_numeric'] == 102.0].shape[0]
                dq_count = player_df[player_df['position_numeric'] == 103.0].shape[0]
                
                if cut_count > 0:
                    logger.info(f"Found {cut_count} players who missed the cut")
                if wd_count > 0:
                    logger.info(f"Found {wd_count} withdrawals")
                if dq_count > 0:
                    logger.info(f"Found {dq_count} disqualifications")
            
            return player_df
        else:
            logger.warning("No player position data could be extracted")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error extracting tournament history: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()
    
    
    
def extract_winners_from_all_tournaments(data_extractor, tournament_ids):
    """Extract winner data from all tournaments."""
    all_results = []
    
    for tournament_id in tournament_ids:
        logger.info(f"Processing tournament {tournament_id}")
        
        # Extract all player data for this tournament
        tournament_data = extract_all_tournament_history(data_extractor, tournament_id)
        
        if not tournament_data.empty:
            # Count winners and other statistics
            winner_count = tournament_data['hist_winner'].sum() if 'hist_winner' in tournament_data else 0
            top3_count = tournament_data['hist_top3'].sum() if 'hist_top3' in tournament_data else 0
            top10_count = tournament_data['hist_top10'].sum() if 'hist_top10' in tournament_data else 0
            
            logger.info(f"Tournament {tournament_id}: Found {winner_count} winners, {top3_count} top3, {top10_count} top10")
            all_results.append(tournament_data)
        else:
            logger.warning(f"No position data extracted for tournament {tournament_id}")
    
    # Combine all tournament results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        logger.info(f"Total combined results: {len(combined_results)} rows")
        return combined_results
    else:
        logger.warning("No tournament data extracted")
        return pd.DataFrame()

def main():
    logger.info("Starting winner extraction from tournament history...")
    
    try:
        # Initialize DB manager and data extractor
        db_manager = DatabaseManager(uri=Config.MONGODB_URI, database_name="pga_tour_data")
        data_extractor = DataExtractor(db_manager)
        logger.info("Initialized database and data extractor")
        
        # Load features file
        logger.info(f"Loading features from {INPUT_FEATURES_PATH}")
        features_df = pd.read_csv(INPUT_FEATURES_PATH)
        logger.info(f"Loaded features with {features_df.shape[0]} rows and {features_df.shape[1]} columns")
        
        # Display some sample column names to help with debugging
        existing_cols = features_df.columns.tolist()
        logger.info(f"Sample existing columns: {existing_cols[:10]}")
        
        # Check for column conflicts
        conflict_cols = [col for col in existing_cols if col in ['winner', 'top3', 'top10', 'top25', 'made_cut']]
        if conflict_cols:
            logger.warning(f"Found potential column conflicts: {conflict_cols}")
        
        # Check data types and convert player_id to string if it's not already
        logger.info(f"player_id data type in features: {features_df['player_id'].dtype}")
        if features_df['player_id'].dtype != 'object':
            logger.info("Converting player_id to string in features dataframe")
            features_df['player_id'] = features_df['player_id'].astype(str)
        
        # Get unique tournament IDs
        tournament_ids = features_df['tournament_id'].unique().tolist()
        logger.info(f"Found {len(tournament_ids)} unique tournaments")
        
        # Extract winner data from tournament history
        winner_df = extract_winners_from_all_tournaments(data_extractor, tournament_ids)
        
        if winner_df.empty:
            logger.warning("No winner data could be extracted from tournament history")
            return
            
        logger.info(f"Extracted winner data with {winner_df.shape[0]} rows")
        logger.info(f"player_id data type in winner data: {winner_df['player_id'].dtype}")
        
        # Ensure player_id is consistently string in winner data
        if winner_df['player_id'].dtype != 'object':
            logger.info("Converting player_id to string in winner dataframe")
            winner_df['player_id'] = winner_df['player_id'].astype(str)
        
        # Get summary stats
        winner_count = winner_df['hist_winner'].sum()
        top3_count = winner_df['hist_top3'].sum()
        top10_count = winner_df['hist_top10'].sum()
        top25_count = winner_df['hist_top25'].sum()
        made_cut_count = winner_df['hist_made_cut'].sum()
        
        logger.info(f"Marked {winner_count} winners with position 1.0")
        logger.info(f"Marked {top3_count} top3 players with positions up to 3.0")
        logger.info(f"Marked {top10_count} top10 players with positions up to 10.0")
        logger.info(f"Marked {top25_count} top25 players with positions up to 25.0")
        logger.info(f"Marked {made_cut_count} players who made the cut")
        
        # Get columns to merge
        merge_columns = ['player_id', 'tournament_id']
        winner_columns = [
            'hist_winner', 'hist_top3', 'hist_top10', 'hist_top25', 
            'hist_made_cut', 'position_numeric', 'original_position'
        ]
        
        columns_to_use = merge_columns + [col for col in winner_columns if col in winner_df.columns]
        
        # Merge with features
        logger.info("Merging winner data with features")
        merged_df = pd.merge(
            features_df,
            winner_df[columns_to_use],
            on=merge_columns,
            how='left'
        )
        
        # Fill NaN values
        for col in ['hist_winner', 'hist_top3', 'hist_top10', 'hist_top25', 'hist_made_cut']:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0).astype(int)
        
        logger.info(f"Final dataset has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
        
        # Calculate match rate
        match_count = merged_df['position_numeric'].notna().sum()
        match_rate = match_count / len(merged_df) * 100
        logger.info(f"Successfully matched position data for {match_count} rows ({match_rate:.1f}%)")
        
        # Save the result
        merged_df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Successfully saved features with winner columns to {OUTPUT_PATH}")
        
    except Exception as e:
        import traceback
        logger.error(f"Error in winner extraction: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()