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
INPUT_FEATURES_PATH = "enhanced_winnerfeatures.csv"  # Update this to your actual file path
OUTPUT_PATH = "features_set_with_winners.csv"  # Output path

# Import your data extractor (adjust import path as needed)
from src.data_excrator.data_excractor import DataExtractor
from database import DatabaseManager
from config import Config

def convert_tournament_id(tournament_id):
    """Convert a tournament ID from RYYYYTTT format to R2025TTT format."""
    if tournament_id.startswith("R") and len(tournament_id) >= 8:
        special_id = "R2025" + tournament_id[5:]
        return special_id
    return tournament_id

def extract_winners_from_history(data_extractor, tournament_ids, player_ids=None):
    """Extract winner data directly from tournament history."""
    result_data = []
    
    for tournament_id in tournament_ids:
        print(f"Processing tournament {tournament_id}")
        
        # Convert to history tournament ID format
        history_tournament_id = convert_tournament_id(tournament_id)
        print(f"Using history ID: {history_tournament_id}")
        
        # Extract tournament history
        history_df = data_extractor.extract_tournament_history(
            tournament_ids=history_tournament_id,
            player_ids=player_ids
        )
        
        if history_df.empty:
            print(f"No history data found for tournament {tournament_id}")
            continue
            
        print(f"Found history data with {len(history_df)} entries")
        
        # Check if position column exists and convert to numeric
        if 'position' in history_df.columns:
            history_df['position_numeric'] = history_df['position'].apply(
                lambda x: pd.to_numeric(str(x).replace('T', ''), errors='coerce') 
                if isinstance(x, (str, int, float)) else None
            )
        
        # Extract position and winner information
        for _, player_row in history_df.iterrows():
            if 'player_id' not in player_row:
                continue
                
            player_id = player_row['player_id']
            
            # Get position
            position = None
            if 'position_numeric' in player_row:
                position = player_row['position_numeric']
            elif 'position' in player_row:
                try:
                    position = float(str(player_row['position']).replace('T', ''))
                except:
                    position = None
            
            if pd.notna(position):
                result_row = {
                    'player_id': player_id,
                    'tournament_id': tournament_id,  # Use original tournament ID
                    'position_numeric': position,
                    'winner': 1 if position == 1 else 0,
                    'top3': 1 if position <= 3 else 0,
                    'top10': 1 if position <= 10 else 0,
                    'top25': 1 if position <= 25 else 0,
                    'made_cut': 1 if position < 100 else 0
                }
                result_data.append(result_row)
    
    # Create DataFrame from results
    if result_data:
        return pd.DataFrame(result_data)
    else:
        return pd.DataFrame()

def main():
    print("Starting winner extraction from tournament history...")
    
    try:
        # Initialize DB manager and data extractor
        db_manager = DatabaseManager(uri=Config.MONGODB_URI, database_name="pga_tour_data")
        data_extractor = DataExtractor(db_manager)
        print("Initialized database and data extractor")
        
        # Load features file
        print(f"Loading features from {INPUT_FEATURES_PATH}")
        features_df = pd.read_csv(INPUT_FEATURES_PATH)
        print(f"Loaded features with {features_df.shape[0]} rows and {features_df.shape[1]} columns")
        
        # Get unique tournament and player IDs
        tournament_ids = features_df['tournament_id'].unique().tolist()
        player_ids = features_df['player_id'].unique().tolist()
        print(f"Found {len(tournament_ids)} unique tournaments and {len(player_ids)} unique players")
        
        # Extract winner data from tournament history
        winner_df = extract_winners_from_history(data_extractor, tournament_ids, player_ids)
        
        if winner_df.empty:
            print("No winner data could be extracted from tournament history")
            return
            
        print(f"Extracted winner data with {winner_df.shape[0]} rows")
        
        # Get summary stats
        winner_count = winner_df['winner'].sum()
        top3_count = winner_df['top3'].sum()
        top10_count = winner_df['top10'].sum()
        top25_count = winner_df['top25'].sum()
        
        print(f"Marked {winner_count} winners with position 1.0")
        print(f"Marked {top3_count} top3 players with positions up to 3.0")
        print(f"Marked {top10_count} top10 players with positions up to 10.0")
        print(f"Marked {top25_count} top25 players with positions up to 25.0")
        
        # Merge with features
        merged_df = pd.merge(
            features_df,
            winner_df[['player_id', 'tournament_id', 'winner', 'top3', 'top10', 'top25', 'made_cut', 'position_numeric']],
            on=['player_id', 'tournament_id'],
            how='left'
        )
        
        # Fill NaN values
        for col in ['winner', 'top3', 'top10', 'top25', 'made_cut']:
            merged_df[col] = merged_df[col].fillna(0).astype(int)
        
        print(f"Final dataset has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
        
        # Save the result
        merged_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Successfully saved features with winner columns to {OUTPUT_PATH}")
        
    except Exception as e:
        import traceback
        print(f"Error in winner extraction: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()