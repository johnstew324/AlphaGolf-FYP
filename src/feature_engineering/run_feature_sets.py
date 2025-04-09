import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import functools
import gc
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Set

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from database import DatabaseManager
from config import Config
from data_excrator.data_excractor import DataExtractor
from feature_engineering.pipeline import FeaturePipeline
from feature_engineering.feature_sets.base_features import create_base_features
from feature_engineering.feature_sets.temporal_features import create_temporal_features
from feature_engineering.feature_sets.interactions_features import create_interaction_features

class ProgressTracker:
    def __init__(self, total_tournaments, total_players):
        self.total_tournaments = total_tournaments
        self.total_players = total_players
        self.processed_tournaments = 0
        self.processed_players = 0
        self.start_time = datetime.now()
        self.tournament_times = {}

        self.output_dir = os.path.join(current_dir, 'output')
        os.makedirs(self.output_dir, exist_ok=True)

        self.completed_file = os.path.join(self.output_dir, 'all_tournaments_features_20250406_190259.json')
        if os.path.exists(self.completed_file):
            with open(self.completed_file, 'r') as f:
                self.completed_tournaments = set(json.load(f))
        else:
            self.completed_tournaments = set()
            
        print(f"Progress tracker initialized. Already completed: {len(self.completed_tournaments)} tournaments")
    
    def update_tournament_progress(self, tournament_id):
        self.processed_tournaments += 1
        self.completed_tournaments.add(tournament_id)
        
        with open(self.completed_file, 'w') as f:
            json.dump(list(self.completed_tournaments), f)
        
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60.0
        remaining = (elapsed / self.processed_tournaments) * (self.total_tournaments - self.processed_tournaments)
        
        print(f"Progress: {self.processed_tournaments}/{self.total_tournaments} tournaments complete")
        print(f"Elapsed time: {elapsed:.1f} minutes")
        print(f"Estimated remaining time: {remaining:.1f} minutes")
    
    def record_tournament_time(self, tournament_id, seconds):
        self.tournament_times[tournament_id] = seconds
        avg_time = sum(self.tournament_times.values()) / len(self.tournament_times)
        print(f"Average tournament processing time: {avg_time:.1f} seconds")
    
    def is_completed(self, tournament_id):
        return tournament_id in self.completed_tournaments

def memoize_with_timeout(timeout=3600):  
    cache = {}
    timestamps = {}
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time.time()

            old_keys = [k for k, ts in timestamps.items() if current_time - ts > timeout]
            for k in old_keys:
                if k in cache:
                    del cache[k]
                if k in timestamps:
                    del timestamps[k]
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = current_time
            return result
        return wrapper
    return decorator

@memoize_with_timeout(timeout=3600)
def cached_extract_tournament_history(data_extractor, tournament_id):
    """Cache tournament history extraction to avoid redundant database calls"""
    print(f"Extracting tournament history for {tournament_id} (cached)")
    return data_extractor.extract_tournament_history(tournament_ids=tournament_id)

@memoize_with_timeout(timeout=3600)
def cached_extract_player_scorecards(data_extractor, tournament_id):
    """Cache player scorecard extraction to avoid redundant database calls"""
    print(f"Extracting player scorecards for {tournament_id} (cached)")
    return data_extractor.extract_player_scorecards(tournament_id)

def get_tournament_players(data_extractor, tournament_id, history_tournament_id, player_id_list=None, batch_size=None):
    """Get players who participated in a tournament with caching"""
    try:
        print(f"Getting players for tournament {tournament_id}...")
        # Use cached function to get tournament history
        tournament_history = cached_extract_tournament_history(data_extractor, history_tournament_id)
        
        tournament_player_ids = []
        
        if not tournament_history.empty:
            if 'players' in tournament_history.columns:
                # Extract player IDs from the players field (if it's a list of player objects)
                for players_list in tournament_history['players']:
                    if isinstance(players_list, list):
                        tournament_player_ids.extend([p.get('player_id') for p in players_list 
                                                 if isinstance(p, dict) and 'player_id' in p])
                
                if tournament_player_ids:
                    print(f"Found {len(tournament_player_ids)} players from tournament history")
            elif 'player_id' in tournament_history.columns:
                # If the history is already flattened by player
                tournament_player_ids = tournament_history['player_id'].unique().tolist()
                print(f"Found {len(tournament_player_ids)} players from flattened tournament history")
        
        # If no players found, try scorecard data
        if not tournament_player_ids:
            print("No player data in tournament history, trying player scorecards...")
            scorecard_data = cached_extract_player_scorecards(data_extractor, tournament_id)
            
            if not scorecard_data.empty and 'player_id' in scorecard_data.columns:
                tournament_player_ids = scorecard_data['player_id'].unique().tolist()
                print(f"Found {len(tournament_player_ids)} players from scorecard data")
        
        # If still no players found, use fallback
        if not tournament_player_ids:
            print("No player data found for this tournament")
            # Use fallback player IDs or the provided player_id_list if available
            tournament_player_ids = player_id_list if player_id_list else ["33948", "35891", "52955", "39971", "39997", "30925"]
            print(f"Using {'provided' if player_id_list else 'fallback'} list of {len(tournament_player_ids)} players")
            return tournament_player_ids
        
        # If we have a player_id_list, filter to only include players in that list
        if player_id_list:
            original_count = len(tournament_player_ids)
            tournament_player_ids = [pid for pid in tournament_player_ids if pid in player_id_list]
            print(f"Filtered from {original_count} to {len(tournament_player_ids)} players from the provided player ID list")
        
        if batch_size and len(tournament_player_ids) > batch_size:
            print(f"Limiting to batch of {batch_size} players from {len(tournament_player_ids)} total")
            # Take a consistent sample to ensure reproducibility
            tournament_player_ids = sorted(tournament_player_ids)[:batch_size]
        
        return tournament_player_ids
            
    except Exception as e:
        print(f"Error getting tournament players: {str(e)}")
        traceback.print_exc()
        # Use fallback player IDs or the provided player_id_list if available
        tournament_player_ids = player_id_list if player_id_list else ["33948", "35891", "52955", "39971", "39997", "30925"]
        print(f"Using {'provided' if player_id_list else 'fallback'} list of {len(tournament_player_ids)} players")
        return tournament_player_ids

def create_player_registry(data_extractor, tournament_id, history_tournament_id, season, player_id_list=None, batch_size=None):
    tournament_player_ids = get_tournament_players(
        data_extractor, tournament_id, history_tournament_id, player_id_list, batch_size
    )
    
    if not tournament_player_ids:
        return pd.DataFrame()
    
    # Create player registry dataframe
    registry = pd.DataFrame({
        'player_id': tournament_player_ids,
        'tournament_id': tournament_id,
        'season': season,
        'history_tournament_id': history_tournament_id
    })
    
    # Duplicate check - make sure each player appears only once
    duplicate_players = registry['player_id'].duplicated().sum()
    if duplicate_players > 0:
        print(f"Warning: Found {duplicate_players} duplicate player entries - removing duplicates")
        registry = registry.drop_duplicates(subset=['player_id'])
    
    return registry

def safely_merge_features(base_df, feature_df, on='player_id', processor_name=None):
    
    if feature_df.empty or on not in feature_df.columns:
        return base_df
    
    # Add processor identifier to columns to avoid conflicts
    if processor_name:
        # Don't rename ID columns or has_* flags
        exclude_cols = [on, 'tournament_id', 'season'] + [col for col in feature_df.columns if col.startswith('has_')]
        rename_cols = {col: f"{col}_{processor_name}" for col in feature_df.columns 
                       if col not in exclude_cols and not col.endswith(f"_{processor_name}")}
        
        if rename_cols:
            feature_df = feature_df.rename(columns=rename_cols)
    
    # Do the merge
    merged = pd.merge(
        base_df,
        feature_df,
        on=on,
        how='left',
        suffixes=('', '_dup')
    )
    
    # Remove duplicate columns
    duplicate_cols = [col for col in merged.columns if col.endswith('_dup')]
    if duplicate_cols:
        merged = merged.drop(columns=duplicate_cols)
    
    return merged

def process_single_tournament_improved(tournament_id, player_id_list=None, player_batch_size=None):
    print(f"\n=== Processing Tournament: {tournament_id} ===")
    tournament_start = datetime.now()

    try:
        db_manager = DatabaseManager(uri=Config.MONGODB_URI, database_name="pga_tour_data")
        data_extractor = DataExtractor(db_manager)
        pipeline = FeaturePipeline(data_extractor)
    except Exception as e:
        print(f"Error initializing components for tournament {tournament_id}: {str(e)}")
        return (tournament_id, None)

    processors = {
        'player_form': pipeline.player_form,
        'course_fit': pipeline.course_fit,
        'tournament_history': pipeline.tournament_history,
        'player_profile': pipeline.player_profile,
        'player_career': pipeline.player_career,
        'scorecard': pipeline.scorecard,
        'tournament_weather': pipeline.tournament_weather,
        'course_stats': pipeline.course_stats,
        'current_form': pipeline.current_form,
        'tournament_history_stats': pipeline.tournament_history_stats
    }

    if tournament_id.startswith("R") and len(tournament_id) >= 8:
        season = int(tournament_id[1:5])
    else:
        season = 2025
        
    print(f"Extracted season: {season}")

    history_tournament_id = tournament_id
    if tournament_id.startswith("R") and len(tournament_id) >= 8:
        tournament_part = tournament_id[5:]

        history_tournament_id = f"R2025{tournament_part}"
        
    print(f"Using tournament history ID: {history_tournament_id}")
    
    # PHASE 1: Create a consistent player registry
    player_registry = create_player_registry(
        data_extractor, 
        tournament_id, 
        history_tournament_id, 
        season, 
        player_id_list, 
        player_batch_size
    )
    
    if player_registry.empty:
        print(f"No players found for tournament {tournament_id}")
        return (tournament_id, None)
    
    # Print player count
    player_count = len(player_registry)
    print(f"Processing {player_count} players for tournament {tournament_id}")
    
    # PHASE 2: Process all feature types using the consistent player registry
    combined_features = None
    
    try:
        # Step 1: Generate base features
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating Base Features for {tournament_id}")
        base_players = player_registry['player_id'].tolist()
        base_features = create_base_features(tournament_id, season, base_players, processors)
        
        if base_features.empty:
            print("No base features generated")
            return (tournament_id, None)
        
        print(f"Successfully generated base features: {base_features.shape[0]} rows, {base_features.shape[1]} columns")
        
        # Start with base features as our combined dataset
        combined_features = base_features.copy()
        
        # Add tournament_id as a column to identify the source tournament
        combined_features['tournament_id'] = tournament_id
        
        # Step 2: Generate temporal features
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating Temporal Features for {tournament_id}")
        temporal_features = create_temporal_features(base_players, tournament_id, season, processors)
        
        if not temporal_features.empty:
            # Merge temporal features
            combined_features = safely_merge_features(
                combined_features, 
                temporal_features, 
                on='player_id',
                processor_name='temporal'
            )
            
            combined_features['has_temporal_features'] = 1
            print(f"Added temporal features. Combined shape: {combined_features.shape}")
        else:
            combined_features['has_temporal_features'] = 0
            print("No temporal features generated")
        
        # Step 3: Generate interaction features
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating Interaction Features for {tournament_id}")
        interaction_features = create_interaction_features(
            base_players, tournament_id, season, processors,
            combined_features, temporal_features
        )
        
        if not interaction_features.empty:
            # Merge interaction features
            combined_features = safely_merge_features(
                combined_features, 
                interaction_features, 
                on='player_id',
                processor_name='interaction'
            )
            
            combined_features['has_interaction_features'] = 1
            print(f"Added interaction features. Combined shape: {combined_features.shape}")
        else:
            combined_features['has_interaction_features'] = 0
            print("No interaction features generated")
        
        # Final check for duplicate players
        if combined_features['player_id'].duplicated().sum() > 0:
            print(f"Warning: Found duplicate players in final dataset. Keeping first occurrence only.")
            combined_features = combined_features.drop_duplicates(subset=['player_id'])
        
        # Calculate data completeness
        has_columns = [col for col in combined_features.columns if col.startswith('has_')]
        if has_columns:
            combined_features['data_completeness'] = combined_features[has_columns].sum(axis=1) / len(has_columns)
        
        print(f"Final combined feature set: {combined_features.shape[0]} rows, {combined_features.shape[1]} columns")
        
        # Calculate elapsed time
        elapsed = (datetime.now() - tournament_start).total_seconds()
        print(f"Tournament {tournament_id} processing completed in {elapsed:.1f} seconds")
        
        # Save individual tournament features
        try:
            output_dir = os.path.join(current_dir, 'output', 'tournaments')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to CSV file
            csv_path = os.path.join(output_dir, f"{tournament_id}_features.csv")
            combined_features.to_csv(csv_path, index=False)
            print(f"Saved tournament features to {csv_path}")
        except Exception as e:
            print(f"Error saving tournament features: {str(e)}")
        
    except Exception as e:
        print(f"Error processing features for tournament {tournament_id}: {str(e)}")
        traceback.print_exc()
    
    # Run garbage collection to free memory
    gc.collect()
    
    return (tournament_id, combined_features)

def extract_position_and_winner_data(data_extractor, tournament_id, player_ids):
    """Extract position and winner data for players in a tournament"""
    print(f"Extracting position and winner data for tournament {tournament_id}")
    
    # Get tournament history data
    tournament_history = data_extractor.extract_tournament_history(
        tournament_ids=tournament_id,
        player_ids=player_ids
    )
    
    if tournament_history.empty:
        print(f"No tournament history found for {tournament_id}")
        return pd.DataFrame()
    
    position_data = []
    
    for _, player_data in tournament_history.iterrows():
        player_info = {
            'player_id': player_data['player_id'],
            'tournament_id': tournament_id
        }

        if 'position' in player_data:
            player_info['position'] = player_data['position']
            
            if isinstance(player_data['position'], str):
                numeric_position = player_data['position'].replace('T', '')
                try:
                    numeric_position = int(numeric_position)
                    player_info['position_numeric'] = numeric_position
                    
                    player_info['is_winner'] = 1 if numeric_position == 1 else 0
                    player_info['is_top3'] = 1 if numeric_position <= 3 else 0
                    player_info['is_top10'] = 1 if numeric_position <= 10 else 0
                    player_info['is_top25'] = 1 if numeric_position <= 25 else 0
                except:
                    player_info['position_numeric'] = None
                    player_info['is_winner'] = 0
                    player_info['is_top3'] = 0
                    player_info['is_top10'] = 0
                    player_info['is_top25'] = 0
            else:
                player_info['position_numeric'] = player_data.get('position_numeric')
                position_value = player_info['position_numeric']
                
                if pd.notna(position_value):
                    player_info['is_winner'] = 1 if position_value == 1 else 0
                    player_info['is_top3'] = 1 if position_value <= 3 else 0
                    player_info['is_top10'] = 1 if position_value <= 10 else 0
                    player_info['is_top25'] = 1 if position_value <= 25 else 0
                else:
                    player_info['is_winner'] = 0
                    player_info['is_top3'] = 0
                    player_info['is_top10'] = 0
                    player_info['is_top25'] = 0

        if 'score_to_par' in player_data:
            player_info['score_to_par'] = player_data['score_to_par']
        
        if 'total_score' in player_data:
            player_info['total_score'] = player_data['total_score']
            
        position_data.append(player_info)
    
    position_df = pd.DataFrame(position_data)
    
    if not position_df.empty and 'is_winner' in position_df.columns:
        winners = position_df[position_df['is_winner'] == 1]
        if not winners.empty:
            winner_id = winners['player_id'].iloc[0]
            position_df['tournament_winner_id'] = winner_id
    
    return position_df

def process_single_tournament_improved(tournament_id, player_id_list=None, player_batch_size=None):
    print(f"\n=== Processing Tournament: {tournament_id} ===")
    tournament_start = datetime.now()

    try:
        db_manager = DatabaseManager(uri=Config.MONGODB_URI, database_name="pga_tour_data")
        data_extractor = DataExtractor(db_manager)
        pipeline = FeaturePipeline(data_extractor)
    except Exception as e:
        print(f"Error initializing components for tournament {tournament_id}: {str(e)}")
        return (tournament_id, None)

    processors = {
        'player_form': pipeline.player_form,
        'course_fit': pipeline.course_fit,
        'tournament_history': pipeline.tournament_history,
        'player_profile': pipeline.player_profile,
        'player_career': pipeline.player_career,
        'scorecard': pipeline.scorecard,
        'tournament_weather': pipeline.tournament_weather,
        'course_stats': pipeline.course_stats,
        'current_form': pipeline.current_form,
        'tournament_history_stats': pipeline.tournament_history_stats
    }

    if tournament_id.startswith("R") and len(tournament_id) >= 8:
        season = int(tournament_id[1:5])
    else:
        season = 2025
        
    print(f"Extracted season: {season}")

    history_tournament_id = tournament_id
    if tournament_id.startswith("R") and len(tournament_id) >= 8:
        tournament_part = tournament_id[5:]
        history_tournament_id = f"R2025{tournament_part}"
        
    print(f"Using tournament history ID: {history_tournament_id}")

    player_registry = create_player_registry(
        data_extractor, 
        tournament_id, 
        history_tournament_id, 
        season, 
        player_id_list, 
        player_batch_size
    )
    
    if player_registry.empty:
        print(f"No players found for tournament {tournament_id}")
        return (tournament_id, None)
    
    player_count = len(player_registry)
    print(f"Processing {player_count} players for tournament {tournament_id}")
    
    combined_features = None
    
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating Base Features for {tournament_id}")
        base_players = player_registry['player_id'].tolist()
        base_features = create_base_features(tournament_id, season, base_players, processors)
        
        if base_features.empty:
            print("No base features generated")
            return (tournament_id, None)
        
        print(f"Successfully generated base features: {base_features.shape[0]} rows, {base_features.shape[1]} columns")
        
        # Start with base features as our combined dataset
        combined_features = base_features.copy()
        
        # Add tournament_id as a column to identify the source tournament
        combined_features['tournament_id'] = tournament_id
        
        # Step 2: Generate temporal features
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating Temporal Features for {tournament_id}")
        temporal_features = create_temporal_features(base_players, tournament_id, season, processors)
        
        if not temporal_features.empty:
            # Merge temporal features
            combined_features = safely_merge_features(
                combined_features, 
                temporal_features, 
                on='player_id',
                processor_name='temporal'
            )
            
            combined_features['has_temporal_features'] = 1
            print(f"Added temporal features. Combined shape: {combined_features.shape}")
        else:
            combined_features['has_temporal_features'] = 0
            print("No temporal features generated")
        
        # Step 3: Generate interaction features
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating Interaction Features for {tournament_id}")
        interaction_features = create_interaction_features(
            base_players, tournament_id, season, processors,
            combined_features, temporal_features
        )
        
        if not interaction_features.empty:
            # Merge interaction features
            combined_features = safely_merge_features(
                combined_features, 
                interaction_features, 
                on='player_id',
                processor_name='interaction'
            )
            
            combined_features['has_interaction_features'] = 1
            print(f"Added interaction features. Combined shape: {combined_features.shape}")
        else:
            combined_features['has_interaction_features'] = 0
            print("No interaction features generated")
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Adding Position and Winner Data for {tournament_id}")
        position_features = extract_position_and_winner_data(
            data_extractor,
            history_tournament_id,  # Use history tournament ID to get past results
            base_players
        )
        
        if not position_features.empty:
            combined_features = safely_merge_features(
                combined_features,
                position_features,
                on=['player_id', 'tournament_id'],
                processor_name='position'
            )
            
            combined_features['has_position_data'] = 1
            print(f"Added position and winner data. Combined shape: {combined_features.shape}")
        else:
            combined_features['has_position_data'] = 0
            print("No position data available")

        if combined_features['player_id'].duplicated().sum() > 0:
            print(f"Warning: Found duplicate players in final dataset. Keeping first occurrence only.")
            combined_features = combined_features.drop_duplicates(subset=['player_id'])
        
        has_columns = [col for col in combined_features.columns if col.startswith('has_')]
        if has_columns:
            combined_features['data_completeness'] = combined_features[has_columns].sum(axis=1) / len(has_columns)
        
        print(f"Final combined feature set: {combined_features.shape[0]} rows, {combined_features.shape[1]} columns")
        
        elapsed = (datetime.now() - tournament_start).total_seconds()
        print(f"Tournament {tournament_id} processing completed in {elapsed:.1f} seconds")

        try:
            output_dir = os.path.join(current_dir, 'output', 'tournaments')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to CSV file
            csv_path = os.path.join(output_dir, f"{tournament_id}_features.csv")
            combined_features.to_csv(csv_path, index=False)
            print(f"Saved tournament features to {csv_path}")
        except Exception as e:
            print(f"Error saving tournament features: {str(e)}")
        
    except Exception as e:
        print(f"Error processing features for tournament {tournament_id}: {str(e)}")
        traceback.print_exc()

    gc.collect()
    
    return (tournament_id, combined_features)


def test_feature_generation_improved(
    tournament_ids, 
    player_id_list=None, 
    tournament_batch_size=5,
    player_batch_size=50,
    max_workers=3,
    resume=True
):
    print(f"\n=== Testing Feature Generation for {len(tournament_ids)} Tournaments ===")
    print(f"Tournament batch size: {tournament_batch_size}")
    print(f"Player batch size: {player_batch_size}")
    print(f"Max workers: {max_workers}")
    
    # Use the enhanced tracker
    tracker = ProgressTracker(
    total_tournaments=len(tournament_ids),
    total_players=len(player_id_list) if player_id_list else 0
    )

    
    if resume:
        original_count = len(tournament_ids)
        tournament_ids = [tid for tid in tournament_ids if not tracker.is_completed(tid)]
        skipped = original_count - len(tournament_ids)
        if skipped > 0:
            print(f"Resuming previous run - skipping {skipped} already completed tournaments")
    
    all_tournament_dfs = []
    
    num_batches = (len(tournament_ids) + tournament_batch_size - 1) // tournament_batch_size
    
    for batch_num in range(num_batches):
        batch_start = batch_num * tournament_batch_size
        batch_end = min(batch_start + tournament_batch_size, len(tournament_ids))
        batch_tournaments = tournament_ids[batch_start:batch_end]
        
        print(f"\n=== Processing Tournament Batch {batch_num+1}/{num_batches} ===")
        print(f"Tournaments in this batch: {batch_tournaments}")
        
        batch_results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_tournament_improved, 
                    tid, 
                    player_id_list,
                    player_batch_size
                ): tid for tid in batch_tournaments
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    tournament_id, tournament_features = future.result()
                    if tournament_features is not None:
                        all_tournament_dfs.append(tournament_features)
                    
                    # Update progress with feature data
                    tracker.update_tournament_progress(tournament_id, tournament_features)
                except Exception as e:
                    print(f"Error processing tournament {tid}: {str(e)}")
                    traceback.print_exc()
        
        # Run garbage collection between batches
        gc.collect()
        
        # Save incremental results
        try:
            if all_tournament_dfs:
                # Create intermediate combined dataframe
                intermediate_df = pd.concat(all_tournament_dfs, ignore_index=True)
                
                # Save to CSV
                output_dir = os.path.join(current_dir, 'output')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(output_dir, f"batch_{batch_num+1}_features.csv")
                intermediate_df.to_csv(csv_path, index=False)
                print(f"\nSaved batch {batch_num+1} features to {csv_path}")
                
                # Print stats
                print(f"Batch stats:")
                print(f"- Total rows: {len(intermediate_df)}")
                print(f"- Unique tournaments: {intermediate_df['tournament_id'].nunique()}")
                print(f"- Unique players: {intermediate_df['player_id'].nunique()}")
                
                # Print winner stats if available
                if 'is_winner' in intermediate_df.columns:
                    winners = intermediate_df[intermediate_df['is_winner'] == 1]
                    print(f"- Winners found: {len(winners)}")
                    print(f"- Tournaments with winner data: {winners['tournament_id'].nunique()}")
        except Exception as e:
            print(f"Error saving incremental results: {str(e)}")
            traceback.print_exc()
    
    # Save position statistics
    tracker.save_position_stats()
    
    combined_df = None
    
    try:
        if all_tournament_dfs:
            # Concatenate all dataframes
            combined_df = pd.concat(all_tournament_dfs, ignore_index=True)
            
            # Check for duplicate player-tournament combinations
            if 'tournament_id' in combined_df.columns and 'player_id' in combined_df.columns:
                dupes = combined_df.duplicated(subset=['tournament_id', 'player_id']).sum()
                if dupes > 0:
                    print(f"Warning: Found {dupes} duplicate player-tournament entries. Removing duplicates.")
                    combined_df = combined_df.drop_duplicates(subset=['tournament_id', 'player_id'])
            
            # Save to a single CSV file
            output_dir = os.path.join(current_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(output_dir, f"all_tournaments_features_{timestamp}.csv")
            combined_df.to_csv(csv_path, index=False)
            print(f"\nSaved all tournament features to {csv_path}")
            
            # Final stats
            print(f"\nFinal dataset stats:")
            print(f"- Total rows: {len(combined_df)}")
            print(f"- Unique tournaments: {combined_df['tournament_id'].nunique()}")
            print(f"- Unique players: {combined_df['player_id'].nunique()}")
            print(f"- Average features per player: {combined_df.shape[1]}")
            
            # Position data stats
            if 'is_winner' in combined_df.columns:
                winners = combined_df[combined_df['is_winner'] == 1]
                print(f"- Total winners found: {len(winners)}")
                print(f"- Tournaments with winner data: {winners['tournament_id'].nunique()}")
            
            if 'position_numeric' in combined_df.columns:
                valid_positions = combined_df['position_numeric'].notna().sum()
                print(f"- Players with position data: {valid_positions}")
                print(f"- Players in top 10: {(combined_df['position_numeric'] <= 10).sum()}")
    except Exception as e:
        print(f"Error saving all tournament features to CSV: {str(e)}")
        traceback.print_exc()
    
    return combined_df

if __name__ == "__main__":
    # Define tournament IDs to test
    tournament_ids = [
            "R2025016", "R2024016", "R2023016", "R2022016", "R2021016", "R2020016", "R2019016", "R2018016",     # "R2017016",# Sentry Tournament of Champions 
            "R2024100", "R2023100", "R2022100", "R2021100", "R2020100", "R2019100", "R2018100",         #"R2017100",# The Open Championship 
            "R2024014", "R2023014", "R2022014", "R2021014", "R2020014", "R2019014", "R2018014",         #"R2017014",# The Masters 
             "R2024033", "R2023033", "R2022033", "R2021033", "R2020033", "R2019033", "R2018033",        #"R2017033",# PGA Championship 
            "R2024026", "R2023026", "R2022026", "R2021026", "R2020026", "R2019026", "R2018026",         #"R2017026",# U.S. Open 
            "R2024007", "R2023007", "R2022007", "R2021007", "R2020007", "R2019007", "R2018007",         #   "R2017007",# genesis invitational 
            "R2024011", "R2023011", "R2022011", "R2021011", "R2020011", "R2019011", "R2018011",             #" R2017011",# THE PLAYERS Championship 
            "R2025012", "R2024012", "R2023012", "R2022012", "R2021012", "R2020012", "R2019012", "R2018012",       #"R2017012",# RBC Heritage 
            "R2024480", "R2024480", "R2023480", "R2022480", "R2021480", "R2020480", "R2019480","R2018480",      #"R2017480",# Wells Fargo Championship
            "R2024023", "R2023023", "R2022023", "R2021023", "R2020023", "R2019023", "R2018023",             #"R2017023",        # Memorial Tournament
            "R2025003", "R2024003", "R2023003", "R2022003", "R2021003", "R2020003", "R2019003","R2018003", #"R2017003",     # waste management phoenix open
            "R2024027", "R2023027", "R2022027", "R2021027", "R2020027", "R2019027", "R2018027", #"R2017027"             #  FedEx St. Jude Championship
            "R2024028", "R2023028", "R2022028", "R2021028", "R2020028", "R2019028", "R2018028", #"R2017028"             #  BMW Championship
            "R2024060", "R2023060", "R2022060", "R2021060", "R2020060", "R2019060", "R2018060", #"R2017060"             #  Tour Championship
            
            #EXTRA
            "R2025006", "R2024006", "R2023006", "R2022006", "R2021006", "R2020006", "R2019006", "R2018006", #"R2017006",#  sony hawai open
            "R2025002", "R2024002", "R2023002", "R2022002", "R2021002", "R2020002", "R2019002", "R2018002",#"R2017002", # american express
            # "R2025475", "R2024475", "R2023475", "R2022475", "R2021475", "R2020475", "R2019475", "R2018475","R2017475", #  Charles Schwab Challenge                                                          
        ]
    
    tournament_file = os.path.join(current_dir, 'tournament_ids.txt')
    if os.path.exists(tournament_file):
        with open(tournament_file, 'r') as f:
            tournament_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(tournament_ids)} tournament IDs from file")
    

    player_ids = ['49303', '33948', '52370', '52955', '22371', '40026', '45522', '56630', '40058', '33141', '34255', '28420', '60004', '48084', '55708', '29268', '47504', '46443', '32070', 
                '35450', '49590', '59440', '34021', '52372', '58933', '32366', '20229', '51766', '47591', '49453', '39997', '59442', '59836', '46435', '32982', '52514', '58605', '39067', '34076', 
                '45157', '28089', '39859', '33653', '51690', '50497', '45609', '59866', 
                '51349', '57362', '47079', '34099', '29725', '54576', '57123', '40098', '30911', '32102', '29936', '29535', '19846', '59018', '52375', '25900', '28679', '55165', '56762', '59095', 
                '51977', '54591', '35310', '31646', '24924', '34563', '33399', '32448', '57975', '47988', '27644', '34098', '49298', '54421', '33597', '60067', '51287', '54628', '51696', '12716', 
                '35532', '52453', '52686', '27141', '39977', '29420', '47056', '35461', '46717', '36801', '35506', '34174', '39971', '36799', '24024', '47917', '34587', '39975', '50188', '37455', 
                '55182', '30926', '29478', '48117', '32757', '47420', '54813', '33122', '36884', '23108', '27936', '28775', '45523', '32791', '37378', '37278', '34409', '36326', '27129', '35449', 
                '40162', '33204', '52215', '34466', '33199', '33413', '32839', '51491', '47993', '59141', '40042', '51950', '28237', '46442', '59143', '32640', '39546', '25198', '55789', '26596', 
                '49947', '50525', '63121', '46601', '36871', '29289', '27349', '35706', '30163', '51070', '51997', '24140', '33968', '55623', '59160', '23320', '20572', '29908', '50582', '48153', 
                '40250', '47679', '36824', '57900', '25818', '49771', '63343', '28252', '34256', '46414', '47983', '26476', '64693', '47995', '36699', '22405', '60882', '39335', '46646', '37275', 
                '56781', '48081', '46046', '47347', '48867', '24502', '46441', '51890', '39327', '29221', '06567', '27139', '46340', '27649', '39324', '34046', '50484', '30692', '51600', '30110', 
                '55893', '49960', '27214', '60019', '40115', '57586', '25493', '51634', '33448', '58168', '32150', '57364', '52144', '30927', '38991', '35617', '32333', '52666', '40006', '27064', 
                '58619', '35658', '54304', '27770', '54607', '25632', '48887', '31113', '27095', '51894', '50474', '32139', '49964', '31323', '08793', '52374', '54783', '54328', '57366', '52513', 
                '45242', '55454', '47483'
    ]
    
    player_file = os.path.join(current_dir, 'player_ids.txt')
    if os.path.exists(player_file):
        with open(player_file, 'r') as f:
            player_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(player_ids)} player IDs from file")
    
    # Run improved feature generation with optimized parameters
    combined_df = test_feature_generation_improved(
        tournament_ids=tournament_ids,
        player_id_list=player_ids,
        tournament_batch_size=5,   
        player_batch_size=50,       
        max_workers=3,             
        resume=True                 
    )
    
    print("\n=== Test Complete ===")
    
    if combined_df is not None:
        print(f"Combined dataset shape: {combined_df.shape}")
        print(f"Number of unique tournaments: {combined_df['tournament_id'].nunique()}")
        print(f"Number of unique players across all tournaments: {combined_df['player_id'].nunique()}")
        
        # Print additional statistics about position data
        print("\n=== Position and Winner Data Statistics ===")
        if 'is_winner' in combined_df.columns:
            winners = combined_df[combined_df['is_winner'] == 1]
            print(f"Number of tournament winners found: {len(winners)}")
            print(f"Percentage of tournaments with winner data: {winners['tournament_id'].nunique() / combined_df['tournament_id'].nunique() * 100:.1f}%")
            
            # Sample of winners
            print("\nSample of tournament winners:")
            winner_sample = winners.sample(min(5, len(winners)))
            for _, row in winner_sample.iterrows():
                tid = row['tournament_id']
                pid = row['player_id']
                name = row.get('player_name', 'Unknown')
                print(f"Tournament {tid}: Winner ID {pid} ({name})")
        
        if 'position_numeric' in combined_df.columns:
            valid_positions = combined_df['position_numeric'].notna()
            top10 = combined_df[valid_positions & (combined_df['position_numeric'] <= 10)]
            print(f"\nPlayers with valid position data: {valid_positions.sum()} ({valid_positions.sum() / len(combined_df) * 100:.1f}%)")
            print(f"Players finishing in top 10: {len(top10)} ({len(top10) / valid_positions.sum() * 100:.1f}% of players with positions)")
            
        print("\n=== Feature Set Sample ===")
        print(combined_df.sample(min(5, len(combined_df))).to_string())