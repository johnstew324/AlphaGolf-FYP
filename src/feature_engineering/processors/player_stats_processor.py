# feature_engineering/processors/player_form_processor.py
import pandas as pd
import numpy as np
from feature_engineering.base import BaseProcessor

class PlayerFormProcessor(BaseProcessor):
    """Process player form features from various data sources."""
    
    def extract_features(self, player_ids, season, tournament_id=None):
        # Extract strokes gained features
        sg_features = self._extract_strokes_gained(player_ids, season)
        
        # If no strokes gained features were found, return empty DataFrame
        if sg_features.empty:
            return pd.DataFrame()
        
        # Extract current form features if tournament_id is provided
        current_form = None
        if tournament_id:
            current_form = self._extract_current_form(tournament_id, player_ids)
        
        # Extract recent performance metrics
        performance = self._extract_performance_metrics(player_ids, season)
        
        # Combine features - start with sg_features
        features = sg_features.copy()
        
        # Add current form features if available
        if current_form is not None and not current_form.empty:
            # Process current form data
            current_form_subset = self._process_current_form(current_form)
            
            # Only attempt to merge if player_id exists in both DataFrames
            if not current_form_subset.empty and 'player_id' in current_form_subset.columns:
                # Merge with features
                features = pd.merge(features, current_form_subset, on='player_id', how='left')
            else:
                print("Warning: Could not merge current form data (missing player_id column)")
        
        # Add performance metrics if available
        if performance is not None and not performance.empty:
            # Only attempt to merge if player_id exists in both DataFrames
            if 'player_id' in performance.columns:
                # Merge performance metrics
                features = pd.merge(features, performance, on='player_id', how='left')
            else:
                print("Warning: Could not merge performance metrics (missing player_id column)")
        
        return features
    
    def _extract_strokes_gained(self, player_ids, season):
        """Extract key strokes gained metrics."""
        player_stats = self.data_extractor.extract_player_stats(
            seasons=season,
            player_ids=player_ids,
            stat_categories=[
                "STROKES_GAINED, SCORING", 
                "STROKES_GAINED, DRIVING",
                "STROKES_GAINED, APPROACH", 
                "STROKES_GAINED, AROUND_GREEN",
                "STROKES_GAINED, PUTTING"
            ]
        )
        
        # Select relevant columns
        if not player_stats.empty:
            # Get only necessary columns
            cols_to_keep = ['player_id', 'name', 'season']
            # Add SG columns
            sg_cols = [col for col in player_stats.columns if 'strokes_gained' in col.lower()]
            
            return player_stats[cols_to_keep + sg_cols]
        else:
            return pd.DataFrame()
    
    def _extract_current_form(self, tournament_id, player_ids):
        """Extract current form data for specific tournament."""
        return self.data_extractor.extract_current_form(
            tournament_id=tournament_id,
            player_ids=player_ids
        )
    
    def _extract_performance_metrics(self, player_ids, season):
        """Extract additional performance metrics."""
        player_stats = self.data_extractor.extract_player_stats(
            seasons=season,
            player_ids=player_ids,
            stat_categories=["SCORING", "DRIVING", "PUTTING", "SCORING"]
        )
        
        if not player_stats.empty:
            # Select key scoring and performance metrics
            cols_to_keep = ['player_id']
            
            # Scoring metrics
            scoring_cols = [
                col for col in player_stats.columns 
                if any(metric in col.lower() for metric in [
                    'scoring_average', 'birdie_average', 'par_breakers',
                    'par_3_scoring', 'par_4_scoring', 'par_5_scoring'
                ])
            ]
            
            # Driving metrics
            driving_cols = [
                col for col in player_stats.columns 
                if any(metric in col.lower() for metric in [
                    'driving_distance', 'driving_accuracy'
                ])
            ]
            
            # Putting metrics
            putting_cols = [
                col for col in player_stats.columns 
                if any(metric in col.lower() for metric in [
                    'putting_average', 'one_putt_percentage', '3_putt_avoidance'
                ])
            ]
            
            return player_stats[cols_to_keep + scoring_cols + driving_cols + putting_cols]
        else:
            return None
    
    def _process_current_form(self, current_form):
        """Process current form data to extract valuable metrics."""
        if current_form.empty:
            return pd.DataFrame()
        
        # Check if player_id exists in the DataFrame
        if 'player_id' not in current_form.columns:
            print("Warning: current_form data does not contain player_id column")
            return pd.DataFrame()
        
        # Keep player_id and valuable features
        cols_to_keep = ['player_id']
        
        # SG metrics from current form
        sg_value_cols = [col for col in current_form.columns if col.endswith('_value') and 'sg_' in col]
        
        # Recent tournament results
        result_cols = [
            col for col in current_form.columns 
            if any(x in col for x in ['position', 'score']) and 'last' in col
        ]
        
        # Ensure all columns exist before selecting
        valid_cols = [col for col in cols_to_keep + sg_value_cols + result_cols if col in current_form.columns]
        
        if set(valid_cols).intersection(set(cols_to_keep)) != set(cols_to_keep):
            # If we lost the player_id column, return empty DataFrame
            return pd.DataFrame()
        
        return current_form[valid_cols]