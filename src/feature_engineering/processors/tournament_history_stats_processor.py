import pandas as pd
import numpy as np
from datetime import datetime
from ..base import BaseProcessor

class TournamentHistoryStatsProcessor(BaseProcessor):
    def extract_features(self, tournament_id=None, player_ids=None, season=None):

        history_stats_df = self.data_extractor.extract_tournament_history_stats(
            tournament_id=tournament_id,
            player_ids=player_ids
        )
        
        if history_stats_df.empty:
            return pd.DataFrame()
    
        features = self._process_tournament_history(history_stats_df, tournament_id)
        
        return features
    
    def _process_tournament_history(self, history_stats_df, tournament_id):
        features = pd.DataFrame()
        if 'player_id' in history_stats_df.columns:
            player_ids = history_stats_df['player_id'].unique()

            player_features = []
            
            for player_id in player_ids:
                player_data = history_stats_df[history_stats_df['player_id'] == player_id]
                
                if player_data.empty:
                    continue
                player_feature = {
                    'player_id': player_id,
                    'tournament_id': tournament_id if tournament_id else player_data['tournament_id'].iloc[0],
                    'total_rounds': player_data['total_rounds'].iloc[0]
                }
                tournament_results = self._extract_tournament_results(player_data)
                player_feature.update(tournament_results)

                strokes_gained = self._extract_strokes_gained(player_data)
                player_feature.update(strokes_gained)
                
                player_features.append(player_feature)

            features = pd.DataFrame(player_features)
        
        return features

    
    def _extract_strokes_gained(self, player_data):
        features = {}
        sg_categories = ['sg_ott', 'sg_app', 'sg_atg', 'sg_p']
        sg_values = {}

        for category in sg_categories:
            col = f"{category}_value"
            if col in player_data.columns:
                try:
                    value = player_data[col].iloc[0]
                    if pd.notna(value):
                        sg_values[category] = float(value)
                except (ValueError, TypeError, IndexError):
                    continue

        if 'sg_ott' in sg_values and 'sg_app' in sg_values:
            features["history_sg_long_game"] = sg_values['sg_ott'] + sg_values['sg_app']
        
        if 'sg_atg' in sg_values and 'sg_p' in sg_values:
            features["history_sg_short_game"] = sg_values['sg_atg'] + sg_values['sg_p']

        if sg_values:
            best_category = max(sg_values, key=sg_values.get)
            worst_category = min(sg_values, key=sg_values.get)
            
            features["history_best_sg_category"] = best_category
            features["history_best_sg_value"] = sg_values[best_category]
            
            features["history_worst_sg_category"] = worst_category
            features["history_worst_sg_value"] = sg_values[worst_category]
            
            features["history_sg_differential"] = sg_values[best_category] - sg_values[worst_category]

        return features
