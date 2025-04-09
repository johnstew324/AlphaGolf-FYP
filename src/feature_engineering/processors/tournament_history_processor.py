# feature_engineering/processors/tournament_history_processor.py
import pandas as pd
import numpy as np
from ..base import BaseProcessor

class TournamentHistoryProcessor(BaseProcessor):
    
    def extract_features(self, tournament_id=None, player_ids=None, season=None):
        # Determine if we're getting player-specific or tournament aggregate features
        get_player_features = player_ids is not None
        
        if get_player_features:
            return self._get_player_tournament_features(tournament_id, player_ids, season)
        else:
            return self._get_tournament_features(tournament_id, season)
    
    def _get_player_tournament_features(self, tournament_id, player_ids, season):

        # Extract raw tournament history data
        history_df = self.data_extractor.extract_tournament_history(
            tournament_ids=tournament_id,
            player_ids=player_ids,
            years=season
        )
        
        if history_df.empty:
            return pd.DataFrame()
        
        # Process the data
        features = self._process_player_history(history_df)
        
        return features
    
    def _get_tournament_features(self, tournament_id, season):
        history_df = self.data_extractor.extract_tournament_history(
            tournament_ids=tournament_id,
            years=season
        )
        
        if history_df.empty:
            return pd.DataFrame()
        
        # Process the data
        features = self._process_tournament_history(history_df)
        
        return features
    
    def _process_player_history(self, history_df):
        df = history_df.copy()

        df['position_numeric'] = df['position'].apply(
            lambda x: pd.to_numeric(x.replace('T', ''), errors='coerce') if isinstance(x, str) else x
        )

        features = pd.DataFrame()
        
        if not df.empty and 'player_id' in df.columns:
            # Group by player and tournament (if tournament_id exists)
            group_cols = ['player_id']
            if 'tournament_id' in df.columns:
                group_cols.append('tournament_id')
            
            grouped = df.groupby(group_cols)
            
            # Calculate aggregate statistics
            features = grouped.agg({
                'position_numeric': ['count', 'mean', 'min', 'max', 'std'],
                'score_to_par': ['mean', 'min', 'max', 'std'],
                'total_score': ['mean', 'min', 'max', 'std'],
                'year': ['min', 'max']
            }).reset_index()
            
            # Flatten multi-index columns
            features.columns = ['_'.join(col).strip('_') for col in features.columns.values]
            
            # Rename columns for clarity
            column_map = {
                'player_id_': 'player_id',
                'tournament_id_': 'tournament_id',
                'position_numeric_count': 'appearances',
                'position_numeric_mean': 'avg_finish',
                'position_numeric_min': 'best_finish',
                'position_numeric_max': 'worst_finish',
                'position_numeric_std': 'finish_std',
                'score_to_par_mean': 'avg_score_to_par',
                'score_to_par_min': 'best_score_to_par',
                'score_to_par_max': 'worst_score_to_par',
                'score_to_par_std': 'score_std',
                'total_score_mean': 'avg_total_score',
                'total_score_min': 'best_total_score',
                'total_score_max': 'worst_total_score',
                'total_score_std': 'total_score_std',
                'year_min': 'first_year_played',
                'year_max': 'last_year_played'
            }
            
            features = features.rename(columns=column_map)
            
            # Calculate additional metrics
            features['cuts_made'] = grouped['position_numeric'].apply(lambda x: x.notnull().sum()).values
            features['cuts_made_pct'] = features['cuts_made'] / features['appearances']
            features['top_10_finishes'] = grouped['position_numeric'].apply(lambda x: (x <= 10).sum()).values
            features['top_25_finishes'] = grouped['position_numeric'].apply(lambda x: (x <= 25).sum()).values
            
            # Add consistency metrics
            features['consistency_ratio'] = features['top_25_finishes'] / features['appearances']
            
            # If we have round scores, calculate round-by-round performance
            if all(col in df.columns for col in ['round1_score', 'round2_score', 'round3_score', 'round4_score']):
                round_cols = [f'round{i}_score' for i in range(1,5)]
                df['rounds_played'] = df[round_cols].notna().sum(axis=1)
                round_stats = df.groupby(group_cols)['rounds_played'].agg(['mean', 'min', 'max']).reset_index()
                round_stats.columns = group_cols + ['avg_rounds_played', 'min_rounds_played', 'max_rounds_played']
                features = pd.merge(features, round_stats, on=group_cols, how='left')
        
        return features
    
    def _process_tournament_history(self, history_df):
        df = history_df.copy()
        
        # Create features
        features = pd.DataFrame()
        
        if not df.empty:
            # Group by tournament to create aggregate features
            grouped = df.groupby('tournament_id')
            
            # Calculate aggregate statistics
            features = grouped.agg({
                'year': ['count', 'min', 'max'],
                'winning_score_to_par': ['mean', 'min', 'max', 'std'],
                'player_count': ['mean', 'min', 'max']
            }).reset_index()
            
            # Flatten multi-index columns
            features.columns = ['_'.join(col).strip('_') for col in features.columns.values]
            
            # Rename columns for clarity
            column_map = {
                'tournament_id_': 'tournament_id',
                'year_count': 'years_recorded',
                'year_min': 'first_year',
                'year_max': 'last_year',
                'winning_score_to_par_mean': 'avg_winning_score',
                'winning_score_to_par_min': 'best_winning_score',
                'winning_score_to_par_max': 'worst_winning_score',
                'winning_score_to_par_std': 'winning_score_std',
                'player_count_mean': 'avg_field_size',
                'player_count_min': 'min_field_size',
                'player_count_max': 'max_field_size'
            }
            
            features = features.rename(columns=column_map)
            
            # Calculate additional metrics
            features['years_span'] = features['last_year'] - features['first_year']
            features['score_variability'] = features['winning_score_std'] / features['avg_winning_score']
            
            # Add winner statistics
            winners = df.groupby(['tournament_id', 'winner_name']).size().reset_index(name='wins')
            top_winners = winners.sort_values(['tournament_id', 'wins'], ascending=[True, False])
            top_winners = top_winners.groupby('tournament_id').head(3)
            
            # Pivot to get top 3 winners per tournament
            top_winners['rank'] = top_winners.groupby('tournament_id').cumcount() + 1
            top_winners_pivot = top_winners.pivot(
                index='tournament_id', 
                columns='rank', 
                values=['winner_name', 'wins']
            )
            
            # Flatten the multi-index columns
            top_winners_pivot.columns = [
                f'{col[0]}_{col[1]}' for col in top_winners_pivot.columns
            ]
            top_winners_pivot = top_winners_pivot.reset_index()
            
            # Merge with features
            features = pd.merge(features, top_winners_pivot, on='tournament_id', how='left')
        
        return features



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
        
        # Create position and winner dataframe
        position_data = []
        
        for _, player_data in tournament_history.iterrows():
            player_info = {
                'player_id': player_data['player_id'],
                'tournament_id': tournament_id
            }
            
            # Extract position information
            if 'position' in player_data:
                player_info['position'] = player_data['position']
                
                # Convert to numeric position (removing 'T' for ties)
                if isinstance(player_data['position'], str):
                    numeric_position = player_data['position'].replace('T', '')
                    try:
                        numeric_position = int(numeric_position)
                        player_info['position_numeric'] = numeric_position
                        
                        # Add winner and top finish flags
                        player_info['is_winner'] = 1 if numeric_position == 1 else 0
                        player_info['is_top3'] = 1 if numeric_position <= 3 else 0
                        player_info['is_top10'] = 1 if numeric_position <= 10 else 0
                        player_info['is_top25'] = 1 if numeric_position <= 25 else 0
                    except:
                        # Handle non-numeric positions like 'CUT', 'WD', etc.
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
            
            # Add additional performance data if available
            if 'score_to_par' in player_data:
                player_info['score_to_par'] = player_data['score_to_par']
            
            if 'total_score' in player_data:
                player_info['total_score'] = player_data['total_score']
                
            position_data.append(player_info)
        
        # Create DataFrame
        position_df = pd.DataFrame(position_data)
        
        # Find the winner
        if not position_df.empty and 'is_winner' in position_df.columns:
            winners = position_df[position_df['is_winner'] == 1]
            if not winners.empty:
                winner_id = winners['player_id'].iloc[0]
                # Add winner_id as a column to all rows
                position_df['tournament_winner_id'] = winner_id
        
        return position_df

# Now let's modify the process_single_tournament_improved function to include this data
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
            
            # NEW STEP: Add position and winner data
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Adding Position and Winner Data for {tournament_id}")
            position_features = extract_position_and_winner_data(
                data_extractor,
                history_tournament_id,  # Use history tournament ID to get past results
                base_players
            )
            
            if not position_features.empty:
                # Merge position features
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