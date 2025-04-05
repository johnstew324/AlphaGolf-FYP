# feature_engineering/processors/tournament_history_stats_processor.py
import pandas as pd
import numpy as np
from datetime import datetime
from ..base import BaseProcessor

class TournamentHistoryStatsProcessor(BaseProcessor):
    """Process tournament history statistics to create meaningful features."""
    
    def extract_features(self, tournament_id=None, player_ids=None, season=None):
        """
        Extract and process tournament history features.
        
        Args:
            tournament_id: Tournament ID to extract
            player_ids: List of player IDs to extract
            season: Season (not directly used for tournament history stats)
            
        Returns:
            DataFrame with processed tournament history features
        """
        # Extract tournament history stats
        history_stats_df = self.data_extractor.extract_tournament_history_stats(
            tournament_id=tournament_id,
            player_ids=player_ids
        )
        
        if history_stats_df.empty:
            return pd.DataFrame()
        
        # Process the data into features
        features = self._process_tournament_history(history_stats_df, tournament_id)
        
        return features
    
    def _process_tournament_history(self, history_stats_df, tournament_id):
        """
        Process tournament history data into meaningful features.
        
        Args:
            history_stats_df: DataFrame with tournament history stats data
            tournament_id: Tournament ID
            
        Returns:
            DataFrame with processed features
        """
        # Create features dataframe
        features = pd.DataFrame()
        
        # Extract player IDs
        if 'player_id' in history_stats_df.columns:
            player_ids = history_stats_df['player_id'].unique()
            
            # Process each player's data
            player_features = []
            
            for player_id in player_ids:
                player_data = history_stats_df[history_stats_df['player_id'] == player_id]
                
                if player_data.empty:
                    continue
                
                # Base player record
                player_feature = {
                    'player_id': player_id,
                    'tournament_id': tournament_id if tournament_id else player_data['tournament_id'].iloc[0],
                    'total_rounds': player_data['total_rounds'].iloc[0]
                }
                
                # Process tournament results
                tournament_results = self._extract_tournament_results(player_data)
                player_feature.update(tournament_results)
                
                # Process strokes gained metrics
                strokes_gained = self._extract_strokes_gained(player_data)
                player_feature.update(strokes_gained)
                
                player_features.append(player_feature)
            
            # Convert to DataFrame
            features = pd.DataFrame(player_features)
        
        return features
    
    def _extract_tournament_results(self, player_data):
        """
        Extract features from historical tournament results.
        
        Args:
            player_data: DataFrame with player's tournament history data
            
        Returns:
            Dictionary of tournament history features
        """
        features = {}
        
        # Find all tournament result columns
        position_cols = [col for col in player_data.columns if col.endswith('_position')]
        score_cols = [col for col in player_data.columns if col.endswith('_score') and 'tournament' in col]
        date_cols = [col for col in player_data.columns if col.endswith('_end_date')]
        
        # If no historical results, return empty features
        if not position_cols:
            features['has_tournament_history'] = 0
            return features
        
        # Calculate position-based features
        features['has_tournament_history'] = 1
        
        # Extract positions and convert to numeric
        positions = []
        for col in position_cols:
            try:
                pos_str = player_data[col].iloc[0]
                if isinstance(pos_str, str):
                    if pos_str.startswith('T'):
                        # Handle tied positions (e.g., "T5")
                        pos = int(pos_str[1:])
                    elif pos_str.isdigit():
                        pos = int(pos_str)
                    elif pos_str.upper() in ['CUT', 'WD', 'DQ']:
                        # Special cases - count as missing cut
                        pos = None
                    else:
                        pos = None
                else:
                    pos = pos_str
                
                positions.append(pos)
            except (ValueError, TypeError, IndexError):
                positions.append(None)
        
        # Filter out None values
        valid_positions = [p for p in positions if p is not None]
        
        if valid_positions:
            # Convert to float for calculations
            valid_positions = [float(p) for p in valid_positions]
            
            # Calculate basic statistics
            features['tournament_appearances'] = len(valid_positions)
            features['avg_finish_position'] = np.mean(valid_positions)
            features['best_finish_position'] = np.min(valid_positions)
            features['worst_finish_position'] = np.max(valid_positions)
            
            if len(valid_positions) > 1:
                features['finish_position_std'] = np.std(valid_positions)
            
            # Calculate achievement counts
            features['wins'] = sum(1 for p in valid_positions if p == 1)
            features['top5'] = sum(1 for p in valid_positions if p <= 5)
            features['top10'] = sum(1 for p in valid_positions if p <= 10)
            features['top25'] = sum(1 for p in valid_positions if p <= 25)
            
            # Calculate percentages if multiple appearances
            if len(valid_positions) > 1:
                features['win_rate'] = features['wins'] / len(valid_positions)
                features['top10_rate'] = features['top10'] / len(valid_positions)
                features['top25_rate'] = features['top25'] / len(valid_positions)
        else:
            # No valid positions
            features['tournament_appearances'] = 0
        
        # Calculate score-based features
        if score_cols:
            scores = []
            for col in score_cols:
                try:
                    score = player_data[col].iloc[0]
                    if pd.notna(score):
                        scores.append(float(score))
                except (ValueError, TypeError, IndexError):
                    pass
            
            if scores:
                features['avg_score'] = np.mean(scores)
                features['best_score'] = np.min(scores)
                features['worst_score'] = np.max(scores)
                
                if len(scores) > 1:
                    features['score_std'] = np.std(scores)
        
        # Calculate recency features
        if date_cols and valid_positions:
            dates = []
            for col in date_cols:
                try:
                    date_val = player_data[col].iloc[0]
                    if pd.notna(date_val):
                        dates.append(pd.to_datetime(date_val))
                except (ValueError, TypeError, IndexError):
                    pass
            
            if dates:
                # Create position-date pairs and sort by date (most recent first)
                position_date_pairs = sorted(
                    zip(valid_positions, dates), 
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Extract most recent position
                features['most_recent_position'] = position_date_pairs[0][0]
                
                # Calculate trend (last 3 appearances)
                if len(position_date_pairs) >= 3:
                    recent_positions = [p for p, _ in position_date_pairs[:3]]
                    
                    # Negative slope means improving (lower position is better)
                    try:
                        x = np.arange(len(recent_positions))
                        slope, _, _, _, _ = np.polyfit(x, recent_positions, 1, full=True)
                        features['recent_trend_slope'] = slope[0]
                    except:
                        pass
                
                # Calculate performance in most recent appearance vs career average
                if len(position_date_pairs) > 1:
                    recent_pos = position_date_pairs[0][0]
                    older_avg = np.mean([p for p, _ in position_date_pairs[1:]])
                    # Positive value means recent performance was worse than average
                    features['recent_vs_history'] = recent_pos - older_avg
        
        return features
    
    def _extract_strokes_gained(self, player_data):
        """
        Extract features from strokes gained metrics.
        
        Args:
            player_data: DataFrame with player's tournament history data
            
        Returns:
            Dictionary of strokes gained features
        """
        features = {}
        
        # Find all strokes gained value columns
        sg_cols = [col for col in player_data.columns if col.endswith('_value') and 'sg_' in col]
        
        if not sg_cols:
            return features
        
        # Standard SG categories
        sg_categories = ['sg_ott', 'sg_app', 'sg_atg', 'sg_p', 'sg_tot']
        
        # Extract each strokes gained category
        for category in sg_categories:
            col = f"{category}_value"
            if col in player_data.columns:
                try:
                    value = player_data[col].iloc[0]
                    if pd.notna(value):
                        features[f"history_{category}"] = float(value)
                except (ValueError, TypeError, IndexError):
                    pass
        
        # Calculate derived metrics
        if all(f"history_{cat}" in features for cat in ['sg_ott', 'sg_app']):
            # Long game (driving + approach)
            features["history_sg_long_game"] = features["history_sg_ott"] + features["history_sg_app"]
        
        if all(f"history_{cat}" in features for cat in ['sg_atg', 'sg_p']):
            # Short game (around green + putting)
            features["history_sg_short_game"] = features["history_sg_atg"] + features["history_sg_p"]
        
        # Calculate contribution percentages
        if 'history_sg_tot' in features and features['history_sg_tot'] != 0:
            for category in ['sg_ott', 'sg_app', 'sg_atg', 'sg_p']:
                if f"history_{category}" in features:
                    features[f"history_{category}_pct"] = (
                        features[f"history_{category}"] / features['history_sg_tot'] * 100
                    )
        
        # Identify strengths and weaknesses at this tournament
        sg_values = {}
        for category in ['sg_ott', 'sg_app', 'sg_atg', 'sg_p']:
            if f"history_{category}" in features:
                sg_values[category] = features[f"history_{category}"]
        
        if sg_values:
            # Find best and worst categories
            best_category = max(sg_values, key=sg_values.get)
            worst_category = min(sg_values, key=sg_values.get)
            
            features["history_best_sg_category"] = best_category
            features["history_best_sg_value"] = sg_values[best_category]
            
            features["history_worst_sg_category"] = worst_category
            features["history_worst_sg_value"] = sg_values[worst_category]
            
            # Calculate skill differential for this tournament
            features["history_sg_differential"] = sg_values[best_category] - sg_values[worst_category]
        
        return features