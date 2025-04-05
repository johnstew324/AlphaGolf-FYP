# feature_engineering/processors/scorecard_processor.py
import pandas as pd
import numpy as np
from ..base import BaseProcessor

class ScorecardProcessor(BaseProcessor):
    """Process player scorecard data to create meaningful features."""
    
    def extract_features(self, tournament_id=None, player_ids=None, season=None):
        """
        Extract and process scorecard features.
        
        Args:
            tournament_id: Tournament ID to extract (standard RYYYY format)
            player_ids: List of player IDs to filter by
            season: Season (typically same as in tournament_id)
            
        Returns:
            DataFrame with processed scorecard features
        """
        # Convert standard tournament_id to format needed by scorecard collection
        # For scorecard data, we use the standard RYYYY format unlike tournament_history
        if tournament_id and tournament_id.startswith("R"):
            scorecard_tournament_id = tournament_id
        else:
            # If somehow a special format was passed, try to convert it
            if tournament_id and tournament_id.startswith("R2025"):
                # Convert from special R2025 format to standard format using season
                year_str = str(season) if season else "2023"  # Default to 2023 if no season
                scorecard_tournament_id = f"R{year_str}{tournament_id[5:]}"
            else:
                scorecard_tournament_id = tournament_id
        
        # Extract hole-by-hole scorecard data for all players
        hole_data = self.data_extractor.extract_player_hole_scores(
            tournament_ids=scorecard_tournament_id,
            player_ids=player_ids
        )
        
        if hole_data.empty:
            return pd.DataFrame()
        
        # Extract round-level scorecard data for all players
        round_data = self.data_extractor.extract_player_scorecards(
            tournament_ids=scorecard_tournament_id,
            player_ids=player_ids
        )
        
        # Process the data into features
        features = self._process_scorecard_data(hole_data, round_data, scorecard_tournament_id)
        
        return features
    
    def _process_scorecard_data(self, hole_data, round_data, tournament_id):
        """
        Process scorecard data to create features.
        
        Args:
            hole_data: DataFrame with hole-by-hole scores
            round_data: DataFrame with round-level scores
            tournament_id: Tournament ID
            
        Returns:
            DataFrame with scorecard-derived features
        """
        # Calculate round statistics for each player
        if not hole_data.empty:
            round_stats = self.data_extractor.calculate_player_round_stats(hole_data)
        else:
            round_stats = pd.DataFrame()
        
        # Create features dataframe
        features = pd.DataFrame()
        
        # If we have round data, use it as the base
        if not round_data.empty:
            # Group by player to get tournament-level aggregates
            player_features = []
            
            for player_id, player_rounds in round_data.groupby('player_id'):
                # Base info
                player_feature = {
                    'player_id': player_id,
                    'tournament_id': tournament_id,
                    'total_rounds_played': len(player_rounds)
                }
                
                # Calculate scoring metrics across all rounds
                player_feature['avg_round_score'] = player_rounds['round_total'].astype(float).mean()
                player_feature['best_round_score'] = player_rounds['round_total'].astype(float).min()
                player_feature['worst_round_score'] = player_rounds['round_total'].astype(float).max()
                player_feature['score_std'] = player_rounds['round_total'].astype(float).std()
                
                if 'score_to_par' in player_rounds.columns:
                    player_feature['avg_score_to_par'] = player_rounds['score_to_par'].astype(float).mean()
                    player_feature['best_score_to_par'] = player_rounds['score_to_par'].astype(float).min()
                    player_feature['worst_score_to_par'] = player_rounds['score_to_par'].astype(float).max()
                
                # Add front nine / back nine stats if available
                if 'front_nine_total' in player_rounds.columns:
                    player_feature['avg_front_nine'] = player_rounds['front_nine_total'].astype(float).mean()
                    player_feature['best_front_nine'] = player_rounds['front_nine_total'].astype(float).min()
                    
                if 'back_nine_total' in player_rounds.columns:
                    player_feature['avg_back_nine'] = player_rounds['back_nine_total'].astype(float).mean()
                    player_feature['best_back_nine'] = player_rounds['back_nine_total'].astype(float).min()
                
                # Calculate round progression metrics
                if len(player_rounds) > 1:
                    # Sort rounds by number
                    sorted_rounds = player_rounds.sort_values('round_number')
                    # Calculate score trend (negative = improving, positive = worsening)
                    if 'score_to_par' in sorted_rounds.columns:
                        scores = sorted_rounds['score_to_par'].astype(float).values
                        player_feature['score_trend'] = np.polyfit(range(len(scores)), scores, 1)[0]
                    
                    # Calculate first round vs. last round difference
                    first_round = sorted_rounds.iloc[0]
                    last_round = sorted_rounds.iloc[-1]
                    
                    if 'score_to_par' in sorted_rounds.columns:
                        player_feature['first_to_last_diff'] = last_round['score_to_par'] - first_round['score_to_par']
                
                player_features.append(player_feature)
            
            # Convert to DataFrame
            features = pd.DataFrame(player_features)
            
            # Add detailed stats from round_stats if available
            if not round_stats.empty:
                # Calculate aggregate stats over all rounds
                agg_stats = round_stats.groupby('player_id').agg({
                    'eagles': 'sum',
                    'birdies': 'sum',
                    'pars': 'sum',
                    'bogeys': 'sum',
                    'double_bogeys': 'sum',
                    'others': 'sum'
                }).reset_index()
                
                # Add par type performance if available
                par_metrics = ['par3_to_par', 'par4_to_par', 'par5_to_par']
                for metric in par_metrics:
                    if metric in round_stats.columns:
                        agg_stats[metric] = round_stats.groupby('player_id')[metric].mean()
                
                # Merge with features
                features = pd.merge(features, agg_stats, on='player_id', how='left')
                
                # Calculate rate stats (per round)
                rate_columns = ['eagles', 'birdies', 'pars', 'bogeys', 'double_bogeys', 'others']
                for col in rate_columns:
                    if col in features.columns:
                        features[f'{col}_per_round'] = features[col] / features['total_rounds_played']
        
        # Add hole-by-hole derived features if available
        if not hole_data.empty:
            # Calculate player consistency metrics
            hole_stats = self._calculate_hole_consistency(hole_data)
            if not hole_stats.empty:
                features = pd.merge(features, hole_stats, on='player_id', how='left')
            
            # Add scoring tendencies by hole type
            hole_type_stats = self._calculate_hole_type_stats(hole_data)
            if not hole_type_stats.empty:
                features = pd.merge(features, hole_type_stats, on='player_id', how='left')
        
        return features
    
    def _calculate_hole_consistency(self, hole_data):
        """
        Calculate player consistency metrics from hole-by-hole data.
        
        Args:
            hole_data: DataFrame with hole-by-hole scores
            
        Returns:
            DataFrame with consistency metrics
        """
        consistency_stats = []
        
        for player_id, player_holes in hole_data.groupby('player_id'):
            # Calculate metrics
            stats = {
                'player_id': player_id,
                'holes_played': len(player_holes),
                'scoring_variability': player_holes['hole_score'].astype(float).std(),
            }
            
            # Calculate streak metrics
            if 'hole_status' in player_holes.columns:
                # Count consecutive good holes (birdie or better)
                status_list = player_holes.sort_values(['round_number', 'hole_number'])['hole_status'].tolist()
                
                good_streaks = []
                current_streak = 0
                for status in status_list:
                    if status in ['EAGLE', 'BIRDIE']:
                        current_streak += 1
                    else:
                        if current_streak > 0:
                            good_streaks.append(current_streak)
                            current_streak = 0
                
                # Add final streak if it exists
                if current_streak > 0:
                    good_streaks.append(current_streak)
                
                stats['max_birdie_streak'] = max(good_streaks) if good_streaks else 0
                stats['avg_birdie_streak'] = sum(good_streaks) / len(good_streaks) if good_streaks else 0
                
                # Count bad streaks (bogey or worse)
                bad_streaks = []
                current_streak = 0
                for status in status_list:
                    if status in ['BOGEY', 'DOUBLE BOGEY', 'TRIPLE BOGEY', 'QUADRUPLE BOGEY']:
                        current_streak += 1
                    else:
                        if current_streak > 0:
                            bad_streaks.append(current_streak)
                            current_streak = 0
                
                # Add final streak if it exists
                if current_streak > 0:
                    bad_streaks.append(current_streak)
                
                stats['max_bogey_streak'] = max(bad_streaks) if bad_streaks else 0
                stats['avg_bogey_streak'] = sum(bad_streaks) / len(bad_streaks) if bad_streaks else 0
            
            consistency_stats.append(stats)
        
        return pd.DataFrame(consistency_stats)
    
    def _calculate_hole_type_stats(self, hole_data):
        """
        Calculate performance stats by hole type (par, length, etc.).
        
        Args:
            hole_data: DataFrame with hole-by-hole scores
            
        Returns:
            DataFrame with hole type stats
        """
        hole_type_stats = []
        
        for player_id, player_holes in hole_data.groupby('player_id'):
            # Base stats
            stats = {
                'player_id': player_id
            }
            
            # Calculate par type performance
            for par in [3, 4, 5]:
                par_holes = player_holes[player_holes['hole_par'] == par]
                if not par_holes.empty:
                    # Convert scores to numeric if needed
                    par_scores = pd.to_numeric(par_holes['hole_score'], errors='coerce')
                    par_par = par_holes['hole_par']
                    
                    # Calculate stats
                    stats[f'par{par}_avg'] = par_scores.mean()
                    stats[f'par{par}_to_par'] = (par_scores - par_par).mean()
                    stats[f'par{par}_holes'] = len(par_holes)
                    
                    # Calculate scoring distribution if hole_status is available
                    if 'hole_status' in par_holes.columns:
                        status_counts = par_holes['hole_status'].value_counts()
                        total = len(par_holes)
                        
                        # Add percentages for each score type
                        for status in ['EAGLE', 'BIRDIE', 'PAR', 'BOGEY', 'DOUBLE BOGEY']:
                            count = status_counts.get(status, 0)
                            stats[f'par{par}_{status.lower().replace(" ", "_")}_pct'] = count / total * 100
            
            # Front nine vs back nine performance
            for nine in ['FRONT', 'BACK']:
                nine_holes = player_holes[player_holes['nine'] == nine]
                if not nine_holes.empty:
                    # Convert scores to numeric if needed
                    nine_scores = pd.to_numeric(nine_holes['hole_score'], errors='coerce')
                    nine_par = nine_holes['hole_par']
                    
                    # Calculate stats
                    lower_nine = nine.lower()
                    stats[f'{lower_nine}_nine_avg'] = nine_scores.mean()
                    stats[f'{lower_nine}_nine_to_par'] = (nine_scores - nine_par).mean()
                    
                    # Calculate scoring distribution if hole_status is available
                    if 'hole_status' in nine_holes.columns:
                        status_counts = nine_holes['hole_status'].value_counts()
                        total = len(nine_holes)
                        
                        # Add percentages for key score types
                        for status in ['BIRDIE', 'PAR', 'BOGEY']:
                            count = status_counts.get(status, 0)
                            stats[f'{lower_nine}_nine_{status.lower()}_pct'] = count / total * 100
            
            hole_type_stats.append(stats)
        
        return pd.DataFrame(hole_type_stats)