import pandas as pd
import numpy as np
from ..base import BaseProcessor

class ScorecardProcessor(BaseProcessor):
    
    def extract_features(self, tournament_id=None, player_ids=None, season=None):
        if tournament_id and tournament_id.startswith("R"):
            scorecard_tournament_id = tournament_id
        else:
            if tournament_id and tournament_id.startswith("R2025"):
                year_str = str(season) if season else "2023"  
                scorecard_tournament_id = f"R{year_str}{tournament_id[5:]}"
            else:
                scorecard_tournament_id = tournament_id
        
        hole_data = self.data_extractor.extract_player_hole_scores(
            tournament_ids=scorecard_tournament_id,
            player_ids=player_ids
        )
        
        if hole_data.empty:
            return pd.DataFrame()
        
        round_data = self.data_extractor.extract_player_scorecards(
            tournament_ids=scorecard_tournament_id,
            player_ids=player_ids
        )
   
        features = self._process_scorecard_data(hole_data, round_data, scorecard_tournament_id)
        
        return features
    
    def _process_scorecard_data(self, hole_data, round_data, tournament_id):
        features = pd.DataFrame()

        if not round_data.empty:
            player_features = []
            
            for player_id, player_rounds in round_data.groupby('player_id'):
                player_feature = {
                    'player_id': player_id,
                    'tournament_id': tournament_id,
                    'total_rounds_played': len(player_rounds)
                }

                player_feature['avg_round_score'] = player_rounds['round_total'].astype(float).mean()
                player_feature['best_round_score'] = player_rounds['round_total'].astype(float).min()
                player_feature['worst_round_score'] = player_rounds['round_total'].astype(float).max()
                
                if 'score_to_par' in player_rounds.columns:
                    player_feature['avg_score_to_par'] = player_rounds['score_to_par'].astype(float).mean()
                    player_feature['best_score_to_par'] = player_rounds['score_to_par'].astype(float).min()

                # Add trends if there's more than one round
                if len(player_rounds) > 1:
                    sorted_rounds = player_rounds.sort_values('round_number')
                    if 'score_to_par' in sorted_rounds.columns:
                        scores = sorted_rounds['score_to_par'].astype(float).values
                        player_feature['score_trend'] = np.polyfit(range(len(scores)), scores, 1)[0]

                player_features.append(player_feature)
            
            features = pd.DataFrame(player_features)
        
        # Add hole-by-hole derived features if available
        if not hole_data.empty:
            hole_stats = self._calculate_hole_consistency(hole_data)
            if not hole_stats.empty:
                features = pd.merge(features, hole_stats, on='player_id', how='left')
            
            hole_type_stats = self._calculate_hole_type_stats(hole_data)
            if not hole_type_stats.empty:
                features = pd.merge(features, hole_type_stats, on='player_id', how='left')
        
        return features
    
    def _calculate_hole_consistency(self, hole_data):
        consistency_stats = []
        
        for player_id, player_holes in hole_data.groupby('player_id'):
            stats = {
                'player_id': player_id,
                'holes_played': len(player_holes),
                'scoring_variability': player_holes['hole_score'].astype(float).std(),
            }
            consistency_stats.append(stats)
        
        return pd.DataFrame(consistency_stats)
    
    def _calculate_hole_type_stats(self, hole_data):
        hole_type_stats = []
        
        for player_id, player_holes in hole_data.groupby('player_id'):
            stats = {
                'player_id': player_id
            }
            
            # Focus on par 3, par 4, and par 5 holes, and reduce the features
            for par in [3, 4, 5]:
                par_holes = player_holes[player_holes['hole_par'] == par]
                if not par_holes.empty:
                    par_scores = pd.to_numeric(par_holes['hole_score'], errors='coerce')
                    par_par = par_holes['hole_par']
                    
                    stats[f'par{par}_avg'] = par_scores.mean()
                    stats[f'par{par}_to_par'] = (par_scores - par_par).mean()
            
            hole_type_stats.append(stats)
        
        return pd.DataFrame(hole_type_stats)
