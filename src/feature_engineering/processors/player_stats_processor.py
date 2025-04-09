import pandas as pd
import numpy as np
from feature_engineering.base import BaseProcessor

class PlayerFormProcessor(BaseProcessor):
    def extract_features(self, player_ids, season, tournament_id=None):

        sg_features = self._extract_strokes_gained(player_ids, season)

        if sg_features.empty:
            return pd.DataFrame()
  
        current_form = None
        if tournament_id:
            current_form = self._extract_current_form(tournament_id, player_ids)
 
        performance = self._extract_performance_metrics(player_ids, season)

        features = sg_features.copy()

        if current_form is not None and not current_form.empty:

            current_form_subset = self._process_current_form(current_form)
            
            if not current_form_subset.empty and 'player_id' in current_form_subset.columns:

                features = pd.merge(features, current_form_subset, on='player_id', how='left')
            else:
                print("Warning: Could not merge current form data (missing player_id column)")
        if performance is not None and not performance.empty:
            if 'player_id' in performance.columns:
                
                features = pd.merge(features, performance, on='player_id', how='left')
            else:
                print("Warning: Could not merge performance metrics (missing player_id column)")
        
        return features
    
    def _extract_strokes_gained(self, player_ids, season):
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
        
        if not player_stats.empty:
            cols_to_keep = ['player_id', 'name', 'season']
            sg_cols = [col for col in player_stats.columns if 'strokes_gained' in col.lower()]
            
            return player_stats[cols_to_keep + sg_cols]
        else:
            return pd.DataFrame()
    
    def _extract_current_form(self, tournament_id, player_ids):
        return self.data_extractor.extract_current_form(
            tournament_id=tournament_id,
            player_ids=player_ids
        )
    
    def _extract_performance_metrics(self, player_ids, season):
        player_stats = self.data_extractor.extract_player_stats(
            seasons=season,
            player_ids=player_ids,
            stat_categories=["SCORING", "DRIVING", "PUTTING", "SCORING"]
        )
        
        if not player_stats.empty:
            cols_to_keep = ['player_id']
            scoring_cols = [
                col for col in player_stats.columns 
                if any(metric in col.lower() for metric in [
                    'scoring_average', 'birdie_average', 'par_breakers',
                    'par_3_scoring', 'par_4_scoring', 'par_5_scoring'
                ])
            ]

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
        if current_form.empty:
            return pd.DataFrame()
        
        if 'player_id' not in current_form.columns:
            print("Warning: current_form data does not contain player_id column")
            return pd.DataFrame()
        
        cols_to_keep = ['player_id']
        
        sg_value_cols = [col for col in current_form.columns if col.endswith('_value') and 'sg_' in col]
        
        result_cols = [
            col for col in current_form.columns 
            if any(x in col for x in ['position', 'score']) and 'last' in col
        ]
        
        valid_cols = [col for col in cols_to_keep + sg_value_cols + result_cols if col in current_form.columns]
        
        if set(valid_cols).intersection(set(cols_to_keep)) != set(cols_to_keep):
            return pd.DataFrame()
        
        return current_form[valid_cols]