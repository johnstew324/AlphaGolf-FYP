# feature_engineering/processors/player_career_processor.py
import pandas as pd
import numpy as np
from ..base import BaseProcessor

class PlayerCareerProcessor(BaseProcessor):
    """Process player career data to create meaningful features."""
    
    def extract_features(self, player_ids=None, season=None, tournament_id=None):
        """
        Extract and process player career features.
        
        Args:
            player_ids: List of player IDs to extract
            season: Optional season filter
            tournament_id: Optional tournament ID (not used for career data)
            
        Returns:
            DataFrame with processed player career features
        """
        # Extract career summary data
        career_df = self.data_extractor.extract_player_career(player_ids=player_ids)
        
        # Extract yearly career data (optionally filtered by season)
        yearly_df = self.data_extractor.extract_player_career_yearly(
            player_ids=player_ids,
            years=season
        )
        
        if career_df.empty and yearly_df.empty:
            return pd.DataFrame()
        
        # Combine features
        features = self._combine_career_data(career_df, yearly_df, season)
        
        return features
    
    def _combine_career_data(self, career_df, yearly_df, season=None):
        """
        Combine career summary and yearly data into a single feature set.
        
        Args:
            career_df: DataFrame from extract_player_career
            yearly_df: DataFrame from extract_player_career_yearly
            season: Optional season filter
            
        Returns:
            Combined DataFrame with player career features
        """
        # Start with career data as base
        if career_df.empty:
            return pd.DataFrame()
        
        features = career_df.copy()
        
        # Add recent performance metrics if available
        if not yearly_df.empty:
            # Get most recent season for each player (or filtered season)
            if season:
                # Use the specified season
                recent_perf = yearly_df[yearly_df['year'] == season]
            else:
                # Get most recent season
                recent_perf = yearly_df.sort_values(['player_id', 'year'], ascending=[True, False])
                recent_perf = recent_perf.groupby('player_id').first().reset_index()
            
            # Select relevant performance columns to merge
            perf_cols = ['player_id']
            stat_cols = [
                'events', 'wins', 'top10', 'top25', 'cuts_made',
                'official_money', 'standings_rank'
            ]
            perf_cols.extend([col for col in stat_cols if col in recent_perf.columns])
            
            # Merge with career data
            features = pd.merge(
                features,
                recent_perf[perf_cols],
                on='player_id',
                how='left',
                suffixes=('', '_current')
            )
            
            # Calculate performance percentages for current season
            if 'cuts_made_current' in features.columns and 'events_current' in features.columns:
                features['current_cut_pct'] = features['cuts_made_current'] / features['events_current']
            
            if 'top10_current' in features.columns and 'events_current' in features.columns:
                features['current_top10_pct'] = features['top10_current'] / features['events_current']
            
            if 'top25_current' in features.columns and 'events_current' in features.columns:
                features['current_top25_pct'] = features['top25_current'] / features['events_current']
        
        # Add career performance aggregates from yearly data
        if not yearly_df.empty:
            career_stats = yearly_df.groupby('player_id').agg({
                'events': 'sum',
                'wins': 'sum',
                'top10': 'sum',
                'top25': 'sum',
                'cuts_made': 'sum',
                'official_money': 'sum'
            }).reset_index()
            
            # Rename columns for career stats
            career_stats = career_stats.rename(columns={
                col: f'career_{col}' for col in career_stats.columns if col != 'player_id'
            })
            
            # Merge with features
            features = pd.merge(
                features,
                career_stats,
                on='player_id',
                how='left'
            )
            
            # Calculate career percentages
            if 'career_cuts_made' in features.columns and 'career_events' in features.columns:
                features['career_cut_pct'] = features['career_cuts_made'] / features['career_events']
            
            if 'career_top10' in features.columns and 'career_events' in features.columns:
                features['career_top10_pct'] = features['career_top10'] / features['career_events']
            
            if 'career_top25' in features.columns and 'career_events' in features.columns:
                features['career_top25_pct'] = features['career_top25'] / features['career_events']
        
        # Process achievement data
        features = self._process_achievement_data(features)
        
        return features
    
    def _process_achievement_data(self, df):
        """
        Process achievement columns to create consistent features.
        
        Args:
            df: DataFrame with raw achievement columns
            
        Returns:
            DataFrame with processed achievement features
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Handle money columns
        money_cols = [col for col in df_processed.columns if 'money' in col.lower() or 'earnings' in col.lower()]
        for col in money_cols:
            if col in df_processed.columns and df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].replace('-', None)
                df_processed[col] = df_processed[col].astype(str).str.replace('[\\$,]', '', regex=True)
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Handle fraction columns (like cuts made)
        fraction_cols = [col for col in df_processed.columns if df_processed[col].astype(str).str.contains('/').any()]
        for col in fraction_cols:
            if col in df_processed.columns:
                fractions = df_processed[col].astype(str).str.split('/', expand=True)
                if len(fractions.columns) == 2:
                    df_processed[f"{col}_numerator"] = pd.to_numeric(fractions[0], errors='coerce')
                    df_processed[f"{col}_denominator"] = pd.to_numeric(fractions[1], errors='coerce')
                    df_processed[f"{col}_pct"] = df_processed[f"{col}_numerator"] / df_processed[f"{col}_denominator"]
        
        # Create consistency features
        if 'career_cut_pct' in df_processed.columns and 'current_cut_pct' in df_processed.columns:
            df_processed['cut_consistency'] = df_processed['current_cut_pct'] - df_processed['career_cut_pct']
        
        if 'career_top10_pct' in df_processed.columns and 'current_top10_pct' in df_processed.columns:
            df_processed['top10_consistency'] = df_processed['current_top10_pct'] - df_processed['career_top10_pct']
        
        if 'career_top25_pct' in df_processed.columns and 'current_top25_pct' in df_processed.columns:
            df_processed['top25_consistency'] = df_processed['current_top25_pct'] - df_processed['career_top25_pct']
        
        return df_processed