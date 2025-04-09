import pandas as pd
import numpy as np
from datetime import datetime
from ..base import BaseProcessor

class CurrentFormProcessor(BaseProcessor):
    def extract_features(self, player_ids=None, season=None, tournament_id=None):
        current_form_df = self.data_extractor.extract_current_form(
            tournament_id=tournament_id,
            player_ids=player_ids
        )
        
        if current_form_df.empty:
            return pd.DataFrame()

        features = self._process_current_form(current_form_df)
        
        return features
    
    def _process_current_form(self, current_form_df):
        features = pd.DataFrame()

        if 'player_id' in current_form_df.columns:
            player_ids = current_form_df['player_id'].unique()

            player_features = []
            
            for player_id in player_ids:
                player_data = current_form_df[current_form_df['player_id'] == player_id]
                
                if player_data.empty:
                    continue
                
                player_feature = {
                    'player_id': player_id,
                    'total_rounds': player_data['total_rounds'].iloc[0]
                }

                if 'tournament_id' in player_data.columns:
                    player_feature['tournament_id'] = player_data['tournament_id'].iloc[0]

                player_feature.update(self._process_recent_results(player_data))
                player_feature.update(self._process_strokes_gained(player_data))
                
                player_features.append(player_feature)

            features = pd.DataFrame(player_features)
        
        return features
    
    def _process_recent_results(self, player_data):
        features = {}

        position_cols = [col for col in player_data.columns if col.endswith('_position')]
        score_cols = [col for col in player_data.columns if col.endswith('_score')]

        if position_cols:
            positions = []
            for col in position_cols:
                try:
                    pos_str = player_data[col].iloc[0]
                    if isinstance(pos_str, str) and pos_str.startswith('T'):
                        pos = int(pos_str[1:])
                    elif pd.notna(pos_str):
                        pos = int(pos_str)
                    else:
                        pos = None
                    
                    positions.append(pos)
                except (ValueError, TypeError):
                    positions.append(None)
            
            positions = [p for p in positions if p is not None]
            
            if positions:
                features['recent_avg_finish'] = np.mean(positions)  # Average finish position
                features['recent_best_finish'] = min(positions)  # Best finish
                features['recent_top5'] = sum(1 for p in positions if p <= 5)  # Top 5 finishes
                features['recent_top10'] = sum(1 for p in positions if p <= 10)  # Top 10 finishes
        
        # Calculate scores
        if score_cols:
            scores = []
            for col in score_cols:
                try:
                    score = float(player_data[col].iloc[0])
                    scores.append(score)
                except (ValueError, TypeError):
                    pass
            
            if scores:
                features['recent_avg_score'] = np.mean(scores)  # Average score to par
                features['recent_best_score'] = min(scores)  # Best score to par
        
        return features
    
    def _process_strokes_gained(self, player_data):
        features = {}
        
        sg_value_cols = [col for col in player_data.columns if col.endswith('_value')]
        
        if sg_value_cols:
            categories = ["sg_ott", "sg_app", "sg_atg", "sg_p", "sg_tot"]
            
            for category in categories:
                col = f"{category}_value"
                if col in player_data.columns:
                    try:
                        value = float(player_data[col].iloc[0])
                        features[category] = value
                    except (ValueError, TypeError):
                        pass
            
            # Calculate long game (SG off the tee + approach)
            if all(f"{cat}_value" in player_data.columns for cat in ["sg_ott", "sg_app"]):
                try:
                    ott = float(player_data["sg_ott_value"].iloc[0])
                    app = float(player_data["sg_app_value"].iloc[0])
                    features["sg_long_game"] = ott + app
                except (ValueError, TypeError):
                    pass
            
            # Calculate short game (SG around green + putting)
            if all(f"{cat}_value" in player_data.columns for cat in ["sg_atg", "sg_p"]):
                try:
                    atg = float(player_data["sg_atg_value"].iloc[0])
                    sg_p = float(player_data["sg_p_value"].iloc[0])
                    features["sg_short_game"] = atg + sg_p
                except (ValueError, TypeError):
                    pass

        return features
