# test_feature_sets.py
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import random

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import components
from database import DatabaseManager
from config import Config
from data_Excator.data_excractor import DataExtractor
from feature_engineering.pipeline import FeaturePipeline
from feature_engineering.feature_sets.base_features import create_base_features
from feature_engineering.feature_sets.temporal_features import create_temporal_features
from feature_engineering.feature_sets.interactions_features import create_interaction_features

def test_feature_generation(tournament_ids, season, player_count=50, specific_player_ids=None):
    """
    Test the generation of feature sets for a list of tournaments.
    
    Args:
        tournament_ids: List of tournament IDs to test (in RYYYYTTT format)
        season: Season year
        player_count: Maximum number of players to include per tournament
        specific_player_ids: Optional list of specific player IDs to test
        
    Returns:
        Dictionary of feature sets by tournament
    """
    print(f"\n=== Testing Feature Generation for {len(tournament_ids)} Tournaments ===")
    print(f"Season: {season}")
    
    # Initialize components
    try:
        print("Connecting to MongoDB...")
        db_manager = DatabaseManager(uri=Config.MONGODB_URI, database_name="pga_tour_data")
        data_extractor = DataExtractor(db_manager)
        pipeline = FeaturePipeline(data_extractor)
        print("Successfully connected to database")
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        return None
    
    # Create a dictionary of processors
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
    
    # Store all feature sets
    all_feature_sets = {}
    
    # Create a list to store all combined feature dataframes
    all_combined_features = []
    
    # Process each tournament
    for tournament_id in tournament_ids:
        print(f"\n=== Processing Tournament: {tournament_id} ===")
        
        # Get player IDs for this tournament if specific ones not provided
        tournament_player_ids = specific_player_ids
        if not tournament_player_ids:
            # Handle the special tournament ID format for tournament_history
            if tournament_id.startswith("R") and len(tournament_id) >= 8:
                special_id = "R2025" + tournament_id[5:]
            else:
                special_id = tournament_id
                
            print(f"Using special ID for tournament history: {special_id}")
            tournament_history = data_extractor.extract_tournament_history(tournament_ids=special_id)
            
            if not tournament_history.empty and 'players' in tournament_history.columns:
                # Extract player IDs from the players field (if it's a list of player objects)
                all_players = []
                for players_list in tournament_history['players']:
                    if isinstance(players_list, list):
                        all_players.extend([p.get('player_id') for p in players_list if isinstance(p, dict) and 'player_id' in p])
                
                if all_players:
                    # Select a random sample of players
                    tournament_player_ids = random.sample(all_players, min(player_count, len(all_players)))
                    print(f"Sampled {len(tournament_player_ids)} players from tournament")
            
            # Alternative: Get players from tournament history processor
            if not tournament_player_ids:
                try:
                    history = pipeline.tournament_history.extract_features(tournament_id=special_id)
                    if not history.empty and 'player_id' in history.columns:
                        tournament_player_ids = history['player_id'].unique().tolist()[:player_count]
                        print(f"Retrieved {len(tournament_player_ids)} players using processor")
                except Exception as e:
                    print(f"Could not get players from tournament history processor: {str(e)}")
            
            # Fallback to specific player IDs if still not found
            if not tournament_player_ids:
                tournament_player_ids = ["33948", "35891", "52955", "39971", "39997", "30925"]
                print(f"Using fallback list of {len(tournament_player_ids)} players")
        
        # Test feature generation for this tournament
        tournament_features = {}
        
        # Step 1: Generate base features
        print("\n--- Generating Base Features ---")
        start_time = datetime.now()
        try:
            base_features = create_base_features(tournament_id, season, tournament_player_ids, processors)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if base_features.empty:
                print("No base features generated")
            else:
                tournament_features['base'] = base_features
                print(f"Successfully generated base features: {base_features.shape[0]} rows, {base_features.shape[1]} columns")
                print(f"Processing time: {elapsed:.2f} seconds")
                
                # Check data availability
                availability_cols = [col for col in base_features.columns if col.startswith('has_')]
                if availability_cols:
                    print("\nData Availability:")
                    for col in availability_cols:
                        available = base_features[col].mean() if not base_features[col].empty else 0
                        print(f"  {col}: {available*100:.1f}%")
        except Exception as e:
            print(f"Error generating base features: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Step 2: Generate temporal features
        print("\n--- Generating Temporal Features ---")
        start_time = datetime.now()
        try:
            temporal_features = create_temporal_features(tournament_player_ids, tournament_id, season, processors)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if temporal_features.empty:
                print("No temporal features generated")
            else:
                tournament_features['temporal'] = temporal_features
                print(f"Successfully generated temporal features: {temporal_features.shape[0]} rows, {temporal_features.shape[1]} columns")
                print(f"Processing time: {elapsed:.2f} seconds")
                
                # Show sample columns
                sample_cols = temporal_features.columns[:5].tolist()
                print(f"\nSample columns: {', '.join(sample_cols)}")
        except Exception as e:
            print(f"Error generating temporal features: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Step 3: Generate interaction features
        print("\n--- Generating Interaction Features ---")
        start_time = datetime.now()
        try:
            # Only proceed if we have base features
            if 'base' in tournament_features:
                temporal = tournament_features.get('temporal', pd.DataFrame())
                interaction_features = create_interaction_features(tournament_player_ids, tournament_id, 
                                                               season, processors,
                                                               tournament_features['base'], temporal)
                elapsed = (datetime.now() - start_time).total_seconds()
                
                if interaction_features.empty:
                    print("No interaction features generated")
                else:
                    tournament_features['interaction'] = interaction_features
                    print(f"Successfully generated interaction features: {interaction_features.shape[0]} rows, {interaction_features.shape[1]} columns")
                    print(f"Processing time: {elapsed:.2f} seconds")
                    
                    # Show sample columns
                    sample_cols = interaction_features.columns[:5].tolist()
                    print(f"\nSample columns: {', '.join(sample_cols)}")
            else:
                print("Skipping interaction features as base features are not available")
        except Exception as e:
            print(f"Error generating interaction features: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Step 4: Create combined feature set
        print("\n--- Creating Combined Feature Set ---")
        try:
            # Only proceed if we have at least base features
            if 'base' in tournament_features:
                combined = tournament_features['base'].copy()
                
                # Add temporal features if available
                if 'temporal' in tournament_features:
                    temporal = tournament_features['temporal']
                    if 'player_id' in temporal.columns:
                        combined = pd.merge(
                            combined,
                            temporal,
                            on='player_id',
                            how='left',
                            suffixes=('', '_temp')
                        )
                        print(f"Added temporal features. Combined shape: {combined.shape}")
                
                # Add interaction features if available
                if 'interaction' in tournament_features:
                    interaction = tournament_features['interaction']
                    if 'player_id' in interaction.columns:
                        combined = pd.merge(
                            combined,
                            interaction,
                            on='player_id',
                            how='left',
                            suffixes=('', '_int')
                        )
                        print(f"Added interaction features. Combined shape: {combined.shape}")
                
                # Remove duplicate columns
                duplicate_cols = [col for col in combined.columns if col.endswith('_temp') or col.endswith('_int')]
                if duplicate_cols:
                    combined = combined.drop(columns=duplicate_cols)
                    print(f"Removed {len(duplicate_cols)} duplicate columns")
                
                # Add tournament ID as a column to identify the source tournament
                combined['tournament_id'] = tournament_id
                
                tournament_features['combined'] = combined
                print(f"Final combined feature set: {combined.shape[0]} rows, {combined.shape[1]} columns")
                
                # Add to the list of all combined features
                all_combined_features.append(combined)
            else:
                print("Skipping combined features as base features are not available")
        except Exception as e:
            print(f"Error creating combined feature set: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Store all feature sets for this tournament
        all_feature_sets[tournament_id] = tournament_features
    
    # Save all tournament features to a single CSV file
    try:
        if all_combined_features:
            # Concatenate all dataframes
            all_tournaments_df = pd.concat(all_combined_features, ignore_index=True)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(current_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to a single CSV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(output_dir, f"all_tournaments_features_{timestamp}.csv")
            all_tournaments_df.to_csv(csv_path, index=False)
            print(f"\nSaved all tournament features to {csv_path}")
    except Exception as e:
        print(f"Error saving all tournament features to CSV: {str(e)}")
    
    return all_feature_sets

def analyze_feature_sets(feature_sets):
    """
    Analyze the generated feature sets.
    
    Args:
        feature_sets: Dictionary of feature sets by tournament
        
    Returns:
        Analysis results
    """
    print("\n=== Analyzing Feature Sets ===")
    
    if not feature_sets:
        print("No feature sets to analyze")
        return {}
    
    analysis = {}
    
    # Analyze each tournament
    for tournament_id, features in feature_sets.items():
        print(f"\nTournament: {tournament_id}")
        
        tournament_analysis = {}
        
        # Analyze each feature type
        for feature_type, feature_df in features.items():
            if feature_df.empty:
                continue
                
            print(f"\n{feature_type.capitalize()} Features:")
            
            # Feature count
            feature_count = feature_df.shape[1]
            tournament_analysis[f'{feature_type}_feature_count'] = feature_count
            print(f"Feature count: {feature_count}")
            
            # Missing values analysis
            missing_pct = feature_df.isnull().mean().mean() * 100
            tournament_analysis[f'{feature_type}_missing_pct'] = missing_pct
            print(f"Average missing values: {missing_pct:.2f}%")
            
            # Categorical features
            cat_cols = feature_df.select_dtypes(include=['object', 'category']).columns
            tournament_analysis[f'{feature_type}_categorical_count'] = len(cat_cols)
            print(f"Categorical features: {len(cat_cols)}")
            
            # Numeric features
            num_cols = feature_df.select_dtypes(include=['int', 'float']).columns
            tournament_analysis[f'{feature_type}_numeric_count'] = len(num_cols)
            print(f"Numeric features: {len(num_cols)}")
            
            # Feature source analysis (from has_* columns)
            availability_cols = [col for col in feature_df.columns if col.startswith('has_')]
            if availability_cols:
                availability = {col: feature_df[col].mean() * 100 for col in availability_cols}
                tournament_analysis[f'{feature_type}_data_sources'] = availability
                print("Data source availability:")
                for source, pct in availability.items():
                    print(f"  {source}: {pct:.1f}%")
        
        analysis[tournament_id] = tournament_analysis
    
    return analysis

def check_feature_overlap(feature_sets):
    """
    Analyze feature overlap between different feature types.
    
    Args:
        feature_sets: Dictionary of feature sets by tournament
        
    Returns:
        Overlap analysis
    """
    print("\n=== Analyzing Feature Overlap ===")
    
    if not feature_sets:
        print("No feature sets to analyze")
        return {}
    
    overlap_analysis = {}
    
    # Analyze one tournament as a sample
    tournament_id = list(feature_sets.keys())[0]
    features = feature_sets[tournament_id]
    
    print(f"\nAnalyzing feature overlap for tournament {tournament_id}")
    
    # Check overlap between base and temporal
    if 'base' in features and 'temporal' in features:
        base_cols = set(features['base'].columns)
        temporal_cols = set(features['temporal'].columns)
        overlap = base_cols.intersection(temporal_cols)
        
        overlap_analysis['base_temporal_overlap'] = {
            'count': len(overlap),
            'percentage': len(overlap) / len(base_cols.union(temporal_cols)) * 100,
            'columns': list(overlap)
        }
        
        print(f"Base-Temporal overlap: {len(overlap)} features ({overlap_analysis['base_temporal_overlap']['percentage']:.1f}%)")
    
    # Check overlap between base and interaction
    if 'base' in features and 'interaction' in features:
        base_cols = set(features['base'].columns)
        interaction_cols = set(features['interaction'].columns)
        overlap = base_cols.intersection(interaction_cols)
        
        overlap_analysis['base_interaction_overlap'] = {
            'count': len(overlap),
            'percentage': len(overlap) / len(base_cols.union(interaction_cols)) * 100,
            'columns': list(overlap)
        }
        
        print(f"Base-Interaction overlap: {len(overlap)} features ({overlap_analysis['base_interaction_overlap']['percentage']:.1f}%)")
    
    # Check overlap between temporal and interaction
    if 'temporal' in features and 'interaction' in features:
        temporal_cols = set(features['temporal'].columns)
        interaction_cols = set(features['interaction'].columns)
        overlap = temporal_cols.intersection(interaction_cols)
        
        overlap_analysis['temporal_interaction_overlap'] = {
            'count': len(overlap),
            'percentage': len(overlap) / len(temporal_cols.union(interaction_cols)) * 100,
            'columns': list(overlap)
        }
        
        print(f"Temporal-Interaction overlap: {len(overlap)} features ({overlap_analysis['temporal_interaction_overlap']['percentage']:.1f}%)")
    
    return overlap_analysis

def profile_tournament_processing(tournament_ids, season, player_ids=None):
    """
    Profile the processing time for different feature types.
    
    Args:
        tournament_ids: List of tournament IDs to test
        season: Season year
        player_ids: Optional list of player IDs to test
        
    Returns:
        Profiling results
    """
    print("\n=== Profiling Tournament Processing ===")
    
    # Initialize components
    try:
        print("Connecting to MongoDB...")
        db_manager = DatabaseManager(uri=Config.MONGODB_URI, database_name="pga_tour_data")
        data_extractor = DataExtractor(db_manager)
        pipeline = FeaturePipeline(data_extractor)
        print("Successfully connected to database")
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        return None
    
    # Create a dictionary of processors
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
    
    profiling_results = {}
    
    # Prepare CSV data structure
    csv_data = []
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test each tournament
    for tournament_id in tournament_ids:
        print(f"\n=== Profiling Tournament: {tournament_id} ===")
        
        # If no player_ids provided, use default test players
        tournament_player_ids = player_ids or ["33948", "35891", "52955", "39971", "39997"]
        
        tournament_profile = {
            'base_features': {},
            'temporal_features': {},
            'interaction_features': {}
        }
        
        # Profile base features
        print("\nProfiling base features...")
        start_time = datetime.now()
        try:
            base_features = create_base_features(tournament_id, season, tournament_player_ids, processors)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            tournament_profile['base_features']['time'] = elapsed
            tournament_profile['base_features']['feature_count'] = base_features.shape[1] if not base_features.empty else 0
            tournament_profile['base_features']['success'] = not base_features.empty
            
            print(f"Base features: {elapsed:.2f} seconds, {tournament_profile['base_features']['feature_count']} features")
        except Exception as e:
            print(f"Error profiling base features: {str(e)}")
            tournament_profile['base_features']['error'] = str(e)
        
        # Profile temporal features
        print("\nProfiling temporal features...")
        start_time = datetime.now()
        try:
            temporal_features = create_temporal_features(tournament_player_ids, tournament_id, season, processors)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            tournament_profile['temporal_features']['time'] = elapsed
            tournament_profile['temporal_features']['feature_count'] = temporal_features.shape[1] if not temporal_features.empty else 0
            tournament_profile['temporal_features']['success'] = not temporal_features.empty
            
            print(f"Temporal features: {elapsed:.2f} seconds, {tournament_profile['temporal_features']['feature_count']} features")
        except Exception as e:
            print(f"Error profiling temporal features: {str(e)}")
            tournament_profile['temporal_features']['error'] = str(e)
        
        # Profile interaction features
        print("\nProfiling interaction features...")
        start_time = datetime.now()
        try:
            interaction_features = create_interaction_features(tournament_player_ids, tournament_id, 
                                                           season, processors,
                                                           base_features, temporal_features)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            tournament_profile['interaction_features']['time'] = elapsed
            tournament_profile['interaction_features']['feature_count'] = interaction_features.shape[1] if not interaction_features.empty else 0
            tournament_profile['interaction_features']['success'] = not interaction_features.empty
            
            print(f"Interaction features: {elapsed:.2f} seconds, {tournament_profile['interaction_features']['feature_count']} features")
        except Exception as e:
            print(f"Error profiling interaction features: {str(e)}")
            tournament_profile['interaction_features']['error'] = str(e)
        
        # Calculate total processing time
        total_time = (tournament_profile['base_features'].get('time', 0) + 
                     tournament_profile['temporal_features'].get('time', 0) +
                     tournament_profile['interaction_features'].get('time', 0))
        tournament_profile['total_time'] = total_time
        tournament_profile['total_feature_count'] = (tournament_profile['base_features'].get('feature_count', 0) +
                                                  tournament_profile['temporal_features'].get('feature_count', 0) +
                                                  tournament_profile['interaction_features'].get('feature_count', 0))
        
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print(f"Total feature count: {tournament_profile['total_feature_count']}")
        
        profiling_results[tournament_id] = tournament_profile
        
        # Add row to CSV data
        csv_row = {
            'timestamp': run_timestamp,
            'tournament_id': tournament_id,
            'season': season,
            'player_count': len(tournament_player_ids),
            'base_time': tournament_profile['base_features'].get('time', 0),
            'base_feature_count': tournament_profile['base_features'].get('feature_count', 0),
            'base_success': tournament_profile['base_features'].get('success', False),
            'temporal_time': tournament_profile['temporal_features'].get('time', 0),
            'temporal_feature_count': tournament_profile['temporal_features'].get('feature_count', 0),
            'temporal_success': tournament_profile['temporal_features'].get('success', False),
            'interaction_time': tournament_profile['interaction_features'].get('time', 0),
            'interaction_feature_count': tournament_profile['interaction_features'].get('feature_count', 0),
            'interaction_success': tournament_profile['interaction_features'].get('success', False),
            'total_time': total_time,
            'total_feature_count': tournament_profile['total_feature_count']
        }
        csv_data.append(csv_row)
    
    # Save results to CSV
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(current_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create or append to CSV file
        csv_file = os.path.join(output_dir, 'profiling_results.csv')
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(csv_file)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, mode='a', header=not file_exists, index=False)
        
        print(f"\nProfiling results saved to {csv_file}")
    except Exception as e:
        print(f"Error saving profiling results to CSV: {str(e)}")
    
    return profiling_results

if __name__ == "__main__":
    # Define test parameters
    tournament_ids = ["R2025016", "R2025011", "R2025007"]  # Sentry Tournament and Masters
    season = 2025
    
    # Optional: Specify player IDs for testing
    player_ids = ["33948", "35891", "52955", "39971", "39997", "28237", "34046", "33448", "46046"]
    
    print("=== Feature Generation Test ===")
    print(f"Testing tournaments: {', '.join(tournament_ids)}")
    print(f"Season: {season}")
    print(f"Testing with {len(player_ids)} players")
    
    # Choose what to test
    run_feature_generation = True  # Changed variable name to avoid conflict
    analyze_features = True
    test_overlap = True
    profile_processing = True
    
    # Run tests
    feature_sets = None
    
    if run_feature_generation:  # Changed variable name here too
        feature_sets = test_feature_generation(tournament_ids, season, player_count=10, specific_player_ids=player_ids)
    
    if analyze_features and feature_sets:
        analysis = analyze_feature_sets(feature_sets)
    
    if test_overlap and feature_sets:
        overlap = check_feature_overlap(feature_sets)
    
    if profile_processing:
        profiling = profile_tournament_processing(tournament_ids, season, player_ids)
    
    print("\n=== Test Complete ===")