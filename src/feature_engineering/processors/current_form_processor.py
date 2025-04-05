# feature_engineering/processors/current_form_processor.py
import pandas as pd
import numpy as np
from datetime import datetime
from ..base import BaseProcessor

class CurrentFormProcessor(BaseProcessor):
    """Process player current form data to create meaningful features."""
    
    def extract_features(self, player_ids=None, season=None, tournament_id=None):
        """
        Extract and process player current form features.
        
        Args:
            player_ids: List of player IDs to extract
            season: Season filter (not directly used for current form)
            tournament_id: Tournament ID to extract form data for
            
        Returns:
            DataFrame with processed current form features
        """
        # Extract current form data
        current_form_df = self.data_extractor.extract_current_form(
            tournament_id=tournament_id,
            player_ids=player_ids
        )
        
        if current_form_df.empty:
            return pd.DataFrame()
        
        # Process the data into features
        features = self._process_current_form(current_form_df)
        
        return features
    
    def _process_current_form(self, current_form_df):
        """
        Process current form data into meaningful features.
        
        Args:
            current_form_df: DataFrame with current form data
            
        Returns:
            DataFrame with processed form features
        """
        # Create features dataframe
        features = pd.DataFrame()
        
        # Extract player IDs
        if 'player_id' in current_form_df.columns:
            player_ids = current_form_df['player_id'].unique()
            
            # Process each player's data
            player_features = []
            
            for player_id in player_ids:
                player_data = current_form_df[current_form_df['player_id'] == player_id]
                
                if player_data.empty:
                    continue
                
                # Base player record
                player_feature = {
                    'player_id': player_id,
                    'total_rounds': player_data['total_rounds'].iloc[0]
                }
                
                # Add tournament ID if available
                if 'tournament_id' in player_data.columns:
                    player_feature['tournament_id'] = player_data['tournament_id'].iloc[0]
                
                # Process recent results
                player_feature.update(self._process_recent_results(player_data))
                
                # Process strokes gained
                player_feature.update(self._process_strokes_gained(player_data))
                
                player_features.append(player_feature)
            
            # Convert to DataFrame
            features = pd.DataFrame(player_features)
        
        return features
    
    def _process_recent_results(self, player_data):
        """
        Process recent tournament results into features.
        
        Args:
            player_data: DataFrame with player data
            
        Returns:
            Dictionary of recent result features
        """
        features = {}
        
        # Find all result columns
        position_cols = [col for col in player_data.columns if col.endswith('_position')]
        score_cols = [col for col in player_data.columns if col.endswith('_score')]
        date_cols = [col for col in player_data.columns if col.endswith('_end_date')]
        
        # Calculate features based on recent positions
        # The current_form data already has the last 5 tournaments
        if position_cols:
            positions = []
            for col in position_cols:
                try:
                    # Handle tied positions (e.g. "T5")
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
            
            # Filter out None values
            positions = [p for p in positions if p is not None]
            
            if positions:
                # Calculate average finish position
                features['recent_avg_finish'] = np.mean(positions)
                
                # Calculate median finish position
                features['recent_median_finish'] = np.median(positions)
                
                # Calculate best and worst finishes
                features['recent_best_finish'] = min(positions)
                features['recent_worst_finish'] = max(positions)
                
                # Calculate consistency (standard deviation of finishes)
                if len(positions) > 1:
                    features['recent_finish_std'] = np.std(positions)
                
                # Calculate top finishes
                features['recent_wins'] = sum(1 for p in positions if p == 1)
                features['recent_top5'] = sum(1 for p in positions if p is not None and p <= 5)
                features['recent_top10'] = sum(1 for p in positions if p is not None and p <= 10)
                features['recent_top25'] = sum(1 for p in positions if p is not None and p <= 25)
                
                # Calculate percentages
                total_events = len(positions)
                if total_events > 0:
                    features['recent_top10_rate'] = features['recent_top10'] / total_events
                    features['recent_top25_rate'] = features['recent_top25'] / total_events
        
        # Calculate features based on recent scores
        if score_cols:
            scores = []
            for col in score_cols:
                try:
                    score = float(player_data[col].iloc[0])
                    scores.append(score)
                except (ValueError, TypeError):
                    pass
            
            if scores:
                # Calculate average score to par
                features['recent_avg_score'] = np.mean(scores)
                
                # Calculate best score to par
                features['recent_best_score'] = min(scores)
                
                # Calculate scoring consistency
                if len(scores) > 1:
                    features['recent_score_std'] = np.std(scores)
        
        # Calculate recency-weighted performance
        if position_cols and len(position_cols) > 1:
            # Get positions in chronological order (most recent first)
            recent_positions = []
            
            for i in range(1, len(position_cols) + 1):
                col = f"last{i}_position"
                if col in player_data.columns:
                    try:
                        pos_str = player_data[col].iloc[0]
                        if isinstance(pos_str, str) and pos_str.startswith('T'):
                            pos = int(pos_str[1:])
                        elif pd.notna(pos_str):
                            pos = int(pos_str)
                        else:
                            pos = None
                        
                        recent_positions.append(pos)
                    except (ValueError, TypeError):
                        recent_positions.append(None)
            
            # Filter out None values
            valid_positions = [p for p in recent_positions if p is not None]
            
            if valid_positions:
                # Apply recency weights (most recent has highest weight)
                weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(valid_positions)]
                
                # Normalize weights to sum to 1
                weights = [w / sum(weights) for w in weights]
                
                # Calculate weighted average position
                features['recent_weighted_finish'] = sum(p * w for p, w in zip(valid_positions, weights))
                
                # Calculate momentum (last tournament compared to previous tournaments)
                if len(valid_positions) >= 2:
                    last_tournament = valid_positions[0]
                    previous_avg = np.mean(valid_positions[1:])
                    features['position_momentum'] = previous_avg - last_tournament  # Positive means improving
        
        # Calculate time since last tournament
        if date_cols and len(date_cols) > 0:
            latest_col = date_cols[0]  # last1_end_date
            try:
                last_date_str = player_data[latest_col].iloc[0]
                if last_date_str:
                    last_date = pd.to_datetime(last_date_str)
                    days_since = (datetime.now() - last_date).days
                    features['days_since_last_tournament'] = days_since
            except:
                pass
        
        return features
    
    def _process_strokes_gained(self, player_data):
        """
        Process strokes gained data into features.
        
        Args:
            player_data: DataFrame with player data
            
        Returns:
            Dictionary of strokes gained features
        """
        features = {}
        
        # Find all strokes gained value columns
        sg_value_cols = [col for col in player_data.columns if col.endswith('_value')]
        
        if sg_value_cols:
            # Extract each strokes gained category
            categories = ["sg_ott", "sg_app", "sg_atg", "sg_p", "sg_tot"]
            
            for category in categories:
                col = f"{category}_value"
                if col in player_data.columns:
                    try:
                        value = float(player_data[col].iloc[0])
                        features[category] = value
                    except (ValueError, TypeError):
                        pass
            
            # Calculate derived metrics
            if all(f"{cat}_value" in player_data.columns for cat in ["sg_ott", "sg_app"]):
                # Long game (driving + approach)
                try:
                    ott = float(player_data["sg_ott_value"].iloc[0])
                    app = float(player_data["sg_app_value"].iloc[0])
                    features["sg_long_game"] = ott + app
                except (ValueError, TypeError):
                    pass
            
            if all(f"{cat}_value" in player_data.columns for cat in ["sg_atg", "sg_p"]):
                # Short game (around green + putting)
                try:
                    atg = float(player_data["sg_atg_value"].iloc[0])
                    sg_p = float(player_data["sg_p_value"].iloc[0])
                    features["sg_short_game"] = atg + sg_p
                except (ValueError, TypeError):
                    pass
                    
            # Calculate strokes gained ratios for consistency
            if "sg_tot_value" in player_data.columns:
                try:
                    sg_tot = float(player_data["sg_tot_value"].iloc[0])
                    
                    # Calculate contribution of each component to total SG
                    for category in ["sg_ott", "sg_app", "sg_atg", "sg_p"]:
                        col = f"{category}_value"
                        if col in player_data.columns and sg_tot != 0:
                            try:
                                value = float(player_data[col].iloc[0])
                                features[f"{category}_contribution"] = value / sg_tot if sg_tot != 0 else 0
                            except (ValueError, TypeError):
                                pass
                except (ValueError, TypeError):
                    pass
            
            # Calculate strengths and weaknesses
            sg_values = {}
            for category in ["sg_ott", "sg_app", "sg_atg", "sg_p"]:
                col = f"{category}_value"
                if col in player_data.columns:
                    try:
                        sg_values[category] = float(player_data[col].iloc[0])
                    except (ValueError, TypeError):
                        sg_values[category] = 0
            
            if sg_values:
                # Find best and worst categories
                best_category = max(sg_values, key=sg_values.get)
                worst_category = min(sg_values, key=sg_values.get)
                
                features["best_sg_category"] = best_category
                features["best_sg_value"] = sg_values[best_category]
                
                features["worst_sg_category"] = worst_category
                features["worst_sg_value"] = sg_values[worst_category]
                
                # Calculate skill polarization (difference between best and worst)
                features["sg_polarization"] = sg_values[best_category] - sg_values[worst_category]
                
                # Calculate skill balance (standard deviation of SG categories)
                if len(sg_values) > 1:
                    features["sg_balance"] = np.std(list(sg_values.values()))
        
        return features