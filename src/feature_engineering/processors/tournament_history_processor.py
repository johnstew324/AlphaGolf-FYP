# feature_engineering/processors/tournament_history_processor.py
import pandas as pd
import numpy as np
from ..base import BaseProcessor

class TournamentHistoryProcessor(BaseProcessor):
    """Process tournament history data to create meaningful features."""
    
    def extract_features(self, tournament_id=None, player_ids=None, season=None):
        """
        Extract and process tournament history features.
        
        Args:
            tournament_id: Tournament ID(s) to extract (uses special R2025 format)
            player_ids: List of player IDs to filter by
            season: Season to filter by (can be used with tournament_id)
            
        Returns:
            DataFrame with processed tournament history features
        """
        # Determine if we're getting player-specific or tournament aggregate features
        get_player_features = player_ids is not None
        
        if get_player_features:
            return self._get_player_tournament_features(tournament_id, player_ids, season)
        else:
            return self._get_tournament_features(tournament_id, season)
    
    def _get_player_tournament_features(self, tournament_id, player_ids, season):
        """
        Get features for specific player-tournament combinations.
        
        Args:
            tournament_id: Tournament ID(s)
            player_ids: List of player IDs
            season: Optional season filter
            
        Returns:
            DataFrame with player-tournament features
        """
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
        """
        Get aggregate features for tournaments.
        
        Args:
            tournament_id: Tournament ID(s)
            season: Optional season filter
            
        Returns:
            DataFrame with tournament-level features
        """
        # Extract raw tournament history data
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
        """
        Process player-specific tournament history to create features.
        
        Args:
            history_df: DataFrame with player tournament history
            
        Returns:
            DataFrame with player-tournament features
        """
        # Make a copy to avoid modifying the original
        df = history_df.copy()
        
        # Convert position to numeric (handling "T1", "CUT", etc.)
        df['position_numeric'] = df['position'].apply(
            lambda x: pd.to_numeric(x.replace('T', ''), errors='coerce') if isinstance(x, str) else x
        )
        
        # Create features
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
        """
        Process general tournament history to create features.
        
        Args:
            history_df: DataFrame with tournament history (aggregate view)
            
        Returns:
            DataFrame with tournament-level features
        """
        # Make a copy to avoid modifying the original
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