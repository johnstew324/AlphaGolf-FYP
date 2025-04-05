# feature_engineering/processors/player_profile_processor.py
import pandas as pd
import numpy as np
from ..base import BaseProcessor

class PlayerProfileProcessor(BaseProcessor):
    """Process player profile overview data to create meaningful features."""
    
    def extract_features(self, player_ids=None, season=None, tournament_id=None):
        """
        Extract and process player profile features.
        
        Args:
            player_ids: List of player IDs to extract
            season: Optional season filter (not used for profile data)
            tournament_id: Optional tournament ID (not used for profile data)
            
        Returns:
            DataFrame with processed player profile features
        """
        # Extract profile overview data
        profile_df = self.data_extractor.extract_player_profile(player_ids=player_ids)
        
        # Extract performance data (most recent season)
        performance_df = self.data_extractor.extract_player_performance(
            player_ids=player_ids,
            tours=['R']  # Focus on PGA Tour
        )
        
        if profile_df.empty and performance_df.empty:
            return pd.DataFrame()
        
        # Combine features
        features = self._combine_profile_data(profile_df, performance_df)
        
        return features
    
    def _combine_profile_data(self, profile_df, performance_df):
        """
        Combine profile overview and performance data into a single feature set.
        
        Args:
            profile_df: DataFrame from extract_player_profile
            performance_df: DataFrame from extract_player_performance
            
        Returns:
            Combined DataFrame with player features
        """
        # Start with profile data as base
        if profile_df.empty:
            return pd.DataFrame()
        
        features = profile_df.copy()
        
        # Add latest performance metrics if available
        if not performance_df.empty:
            # Get most recent season for each player
            latest_perf = performance_df.sort_values(['player_id', 'season'], ascending=[True, False])
            latest_perf = latest_perf.groupby('player_id').first().reset_index()
            
            # Select relevant performance columns to merge
            perf_cols = ['player_id']
            stat_cols = [
                'events', 'cuts_made', 'wins', 'seconds', 'thirds',
                'top_10', 'top_25', 'earnings'
            ]
            perf_cols.extend([col for col in stat_cols if col in latest_perf.columns])
            
            # Merge with profile data
            features = pd.merge(
                features,
                latest_perf[perf_cols],
                on='player_id',
                how='left',
                suffixes=('', '_current')
            )
            
            # Calculate performance percentages
            if 'cuts_made' in features.columns and 'events' in features.columns:
                features['cuts_made_pct'] = features['cuts_made'] / features['events']
            
            if 'top_10' in features.columns and 'events' in features.columns:
                features['top_10_pct'] = features['top_10'] / features['events']
            
            if 'top_25' in features.columns and 'events' in features.columns:
                features['top_25_pct'] = features['top_25'] / features['events']
        
        # Add career performance aggregates if available
        if not performance_df.empty:
            career_stats = performance_df.groupby('player_id').agg({
                'events': 'sum',
                'cuts_made': 'sum',
                'wins': 'sum',
                'seconds': 'sum',
                'thirds': 'sum',
                'top_10': 'sum',
                'top_25': 'sum',
                'earnings': 'sum'
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
                features['career_cuts_made_pct'] = features['career_cuts_made'] / features['career_events']
            
            if 'career_top_10' in features.columns and 'career_events' in features.columns:
                features['career_top_10_pct'] = features['career_top_10'] / features['career_events']
            
            if 'career_top_25' in features.columns and 'career_events' in features.columns:
                features['career_top_25_pct'] = features['career_top_25'] / features['career_events']
        
        return features
    
    def _process_snapshot_data(self, snapshot_data):
        """
        Process the snapshot data from player profiles.
        
        Args:
            snapshot_data: List of snapshot items from profile data
            
        Returns:
            Dictionary of processed snapshot features
        """
        snapshot_features = {}
        
        for item in snapshot_data:
            title = item.get("title", "").lower().replace(" ", "_")
            value = item.get("value", "")
            description = item.get("description", "")
            
            # Add main value
            snapshot_features[f"snapshot_{title}"] = value
            
            # Add description if it exists
            if description:
                snapshot_features[f"snapshot_{title}_desc"] = description
            
            # Special handling for specific snapshot items
            if title == "lowest_round":
                # Extract just the score number if available
                try:
                    score = int(value) if value.isdigit() else None
                    snapshot_features["lowest_round_score"] = score
                except (ValueError, AttributeError):
                    pass
        
        return snapshot_features
    
    def _process_standings_data(self, standings_data):
        """
        Process standings data from player profiles.
        
        Args:
            standings_data: Dictionary of standings data
            
        Returns:
            Dictionary of processed standings features
        """
        standings_features = {}
        
        if standings_data:
            for key, value in standings_data.items():
                if key not in ['title', 'description', 'detail_copy']:
                    standings_features[f"standings_{key}"] = value
        
        return standings_features
    
    def _process_fedex_fall_data(self, fedex_data):
        """
        Process FedEx Fall standings data from player profiles.
        
        Args:
            fedex_data: Dictionary of FedEx Fall data
            
        Returns:
            Dictionary of processed FedEx Fall features
        """
        fedex_features = {}
        
        if fedex_data:
            for key, value in fedex_data.items():
                if key not in ['title', 'description', 'detail_copy']:
                    fedex_features[f"fedex_fall_{key}"] = value
        
        return fedex_features
    

# to do to test!!!!!!
    def _process_owgr_data(self, profile_data):
        """
        Process OWGR (Official World Golf Ranking) data from player profiles.
        
        Args:
            profile_data: DataFrame with profile data
            
        Returns:
            DataFrame with processed OWGR features
        """
        # Start with player_id column
        if 'player_id' not in profile_data.columns or profile_data.empty:
            return pd.DataFrame()
            
        owgr_features = profile_data[['player_id']].copy()
        
        # Process standard OWGR
        owgr_columns = ['standings_owgr', 'debug_owgr', 'owgr']
        owgr_found = False
        
        for col in owgr_columns:
            if col in profile_data.columns and not profile_data[col].isna().all():
                owgr_features['owgr'] = pd.to_numeric(profile_data[col], errors='coerce')
                owgr_found = True
                break
        
        # If we found OWGR, create additional features
        if owgr_found:
            # Create OWGR tiers
            conditions = [
                (owgr_features['owgr'] <= 10),
                (owgr_features['owgr'] <= 25) & (owgr_features['owgr'] > 10),
                (owgr_features['owgr'] <= 50) & (owgr_features['owgr'] > 25),
                (owgr_features['owgr'] <= 100) & (owgr_features['owgr'] > 50),
                (owgr_features['owgr'] > 100)
            ]
            
            tier_labels = ['Elite', 'Top 25', 'Top 50', 'Top 100', 'Outside 100']
            owgr_features['owgr_tier'] = np.select(conditions, tier_labels, default='Unknown')
            
            # Create inverted OWGR for use in calculations (higher is better)
            # Using a log scale helps prevent outliers for very low-ranked players
            owgr_features['owgr_score'] = 1000 - (100 * np.log10(owgr_features['owgr']))
            
            # Normalize to 0-100 scale for easier interpretation
            if len(owgr_features) > 1:  # Only normalize if we have multiple players
                max_score = owgr_features['owgr_score'].max()
                min_score = owgr_features['owgr_score'].min()
                if max_score > min_score:
                    owgr_features['owgr_score_norm'] = (
                        (owgr_features['owgr_score'] - min_score) / (max_score - min_score) * 100
                    )
            else:
                # For single player, use a reference scale
                owgr_features['owgr_score_norm'] = 100 - (owgr_features['owgr'] / 200 * 100).clip(0, 100)
        
        return owgr_features