# feature_engineering/feature_sets/temporal_features.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def create_recent_performance_features(player_ids: List[str], tournament_id: str, 
                                      processors: Dict, window_size: int = 5) -> pd.DataFrame:

    if 'current_form' in processors:
        current_form = processors['current_form'].extract_features(
            player_ids=player_ids,
            tournament_id=tournament_id
        )
        
        if not current_form.empty:
            # Process to extract temporal patterns
            temporal_features = _process_recent_form(current_form, window_size)
            return temporal_features
    
    # If current_form not available, return empty DataFrame
    return pd.DataFrame({'player_id': player_ids})

def create_consistency_features(player_ids: List[str], season: int, 
                              processors: Dict, window_size: int = 5) -> pd.DataFrame:

    if 'tournament_history' in processors:
        if 'current_form' in processors:
            current_form = processors['current_form'].extract_features(
                player_ids=player_ids
            )
            
            if not current_form.empty:
                # Calculate consistency metrics
                consistency_features = _calculate_consistency_metrics(current_form)
                return consistency_features
    
    # If data not available, return empty DataFrame
    return pd.DataFrame({'player_id': player_ids})

def create_momentum_features(player_ids: List[str], tournament_id: str, 
                           processors: Dict, window_size: int = 5) -> pd.DataFrame:

    if 'current_form' in processors:
        current_form = processors['current_form'].extract_features(
            player_ids=player_ids,
            tournament_id=tournament_id
        )
        
        if not current_form.empty:
            # Calculate momentum metrics
            momentum_features = _calculate_momentum_metrics(current_form)
            return momentum_features
    
    # If current_form not available, return empty DataFrame
    return pd.DataFrame({'player_id': player_ids})

def create_strokes_gained_trends(player_ids: List[str], season: int, 
                               processors: Dict) -> pd.DataFrame:
 
    if 'current_form' in processors:
        current_form = processors['current_form'].extract_features(
            player_ids=player_ids
        )
        
        if not current_form.empty:
            # Calculate SG trends
            sg_trends = _calculate_sg_trends(current_form)
            return sg_trends
    
    # If current_form not available, return empty DataFrame
    return pd.DataFrame({'player_id': player_ids})

def create_temporal_features(player_ids: List[str], tournament_id: str, 
                           season: int, processors: Dict) -> pd.DataFrame:
   
    recent_perf = create_recent_performance_features(player_ids, tournament_id, processors)
    consistency = create_consistency_features(player_ids, season, processors)
    momentum = create_momentum_features(player_ids, tournament_id, processors)
    sg_trends = create_strokes_gained_trends(player_ids, season, processors)
    
    # Combine all features
    all_features = []
    
    # Start with player IDs if all features are empty
    if (recent_perf.empty and consistency.empty and 
        momentum.empty and sg_trends.empty):
        return pd.DataFrame({'player_id': player_ids})
    
    # Add features that have data
    if not recent_perf.empty:
        all_features.append(recent_perf)
    
    if not consistency.empty:
        all_features.append(consistency)
    
    if not momentum.empty:
        all_features.append(momentum)
    
    if not sg_trends.empty:
        all_features.append(sg_trends)
    
    # Merge all feature sets
    temporal_features = _merge_feature_sets(all_features, on='player_id')
    
    # Add availability metadata
    temporal_features['has_temporal_features'] = 1
    
    return temporal_features

def _process_recent_form(current_form: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
   
    # Start with player_id column
    if 'player_id' not in current_form.columns:
        return pd.DataFrame()
        
    features = current_form[['player_id']].copy()
    
    # Find position columns (last1_position, last2_position, etc.)
    position_cols = [col for col in current_form.columns if col.endswith('_position')]
    position_cols = sorted(position_cols)[:window_size]  # Limit to window_size
    
    # Find score columns
    score_cols = [col for col in current_form.columns if col.endswith('_score')]
    score_cols = sorted(score_cols)[:window_size]  # Limit to window_size
    
    # Count tournaments with data available
    if position_cols:
        # Convert position strings (e.g., "T5") to numeric
        positions = pd.DataFrame()
        for col in position_cols:
            positions[col] = current_form[col].apply(
                lambda x: int(x.replace('T', '')) if isinstance(x, str) and x.startswith('T') 
                else int(x) if pd.notna(x) and str(x).isdigit() 
                else np.nan
            )
        
        # Calculate number of valid tournaments
        features['valid_tournaments'] = positions.notna().sum(axis=1)
        
        # Only proceed if we have enough valid tournaments
        mask = features['valid_tournaments'] > 0
        
        if mask.any():
            # Calculate simple moving average of positions
            features.loc[mask, 'avg_recent_position'] = positions.loc[mask].mean(axis=1)
            
            # Calculate position trend (positive = worsening, negative = improving)
            if len(position_cols) >= 2:
                # Use simple linear regression slope to measure trend
                for i, player_id in enumerate(features.loc[mask, 'player_id']):
                    player_positions = positions.loc[positions.index[mask][i]]
                    valid_positions = player_positions.dropna()
                    
                    if len(valid_positions) >= 2:
                        # Get indices (0,1,2,...) for valid positions and convert to list
                        indices = np.arange(len(valid_positions)).reshape(-1, 1)
                        position_values = valid_positions.values.reshape(-1, 1)
                        
                        # Calculate slope using numpy's polyfit
                        if len(indices) > 0 and len(position_values) > 0:
                            slope = np.polyfit(indices.flatten(), position_values.flatten(), 1)[0]
                            features.loc[features.index[mask][i], 'position_trend'] = slope
    
    # Process score columns if available
    if score_cols:
        # Extract numeric scores
        scores = pd.DataFrame()
        for col in score_cols:
            scores[col] = pd.to_numeric(current_form[col], errors='coerce')
        
        # Calculate average score (lower is better)
        mask = scores.notna().any(axis=1)
        if mask.any():
            features.loc[mask, 'avg_recent_score'] = scores.loc[mask].mean(axis=1)
            
            # Calculate score trend (negative = improving, positive = worsening)
            if len(score_cols) >= 2:
                for i, player_id in enumerate(features.loc[mask, 'player_id']):
                    player_scores = scores.loc[scores.index[mask][i]]
                    valid_scores = player_scores.dropna()
                    
                    if len(valid_scores) >= 2:
                        indices = np.arange(len(valid_scores)).reshape(-1, 1)
                        score_values = valid_scores.values.reshape(-1, 1)
                        
                        if len(indices) > 0 and len(score_values) > 0:
                            slope = np.polyfit(indices.flatten(), score_values.flatten(), 1)[0]
                            features.loc[features.index[mask][i], 'score_trend'] = slope
    
    # Add weighted recency score (more weight to recent tournaments)
    if position_cols and len(position_cols) > 0:
        # Create weights with more emphasis on recent tournaments
        weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(position_cols)]
        weights = [w / sum(weights) for w in weights]  # Normalize weights
        
        # Calculate weighted position average
        for i, player_id in enumerate(features['player_id']):
            player_positions = positions.loc[i, position_cols].copy()
            valid_mask = player_positions.notna()
            
            if valid_mask.any():
                # Get valid positions and corresponding weights
                valid_positions = player_positions[valid_mask].values
                valid_weights = [weights[j] for j, is_valid in enumerate(valid_mask) if is_valid]
                
                # Normalize valid weights
                valid_weights = [w / sum(valid_weights) for w in valid_weights]
                
                # Calculate weighted average
                weighted_avg = sum(p * w for p, w in zip(valid_positions, valid_weights))
                features.loc[i, 'weighted_recent_position'] = weighted_avg
    
    return features

def _calculate_consistency_metrics(current_form: pd.DataFrame) -> pd.DataFrame:
  
    # Start with player_id column
    if 'player_id' not in current_form.columns:
        return pd.DataFrame()
        
    features = current_form[['player_id']].copy()
    
    # Find position columns (last1_position, last2_position, etc.)
    position_cols = [col for col in current_form.columns if col.endswith('_position')]
    position_cols = sorted(position_cols)
    
    if position_cols:
        # Convert position strings to numeric
        positions = pd.DataFrame()
        for col in position_cols:
            positions[col] = current_form[col].apply(
                lambda x: int(x.replace('T', '')) if isinstance(x, str) and x.startswith('T') 
                else int(x) if pd.notna(x) and str(x).isdigit() 
                else np.nan
            )
        
        # Calculate consistency metrics
        mask = positions.notna().sum(axis=1) > 1  # Need at least 2 tournaments
        
        if mask.any():
            # Standard deviation of positions (lower = more consistent)
            features.loc[mask, 'position_std'] = positions.loc[mask].std(axis=1)
            
            # Coefficient of variation (normalized std dev)
            position_means = positions.loc[mask].mean(axis=1)
            position_stds = positions.loc[mask].std(axis=1)
            features.loc[mask, 'position_cv'] = position_stds / position_means
            
            # Range of positions (max - min)
            features.loc[mask, 'position_range'] = positions.loc[mask].max(axis=1) - positions.loc[mask].min(axis=1)
            
            # Count of finishes in different tiers
            for i, player_id in enumerate(features.loc[mask, 'player_id']):
                player_positions = positions.loc[positions.index[mask][i]]
                valid_positions = player_positions.dropna()
                
                if len(valid_positions) > 0:
                    features.loc[features.index[mask][i], 'top10_count'] = (valid_positions <= 10).sum()
                    features.loc[features.index[mask][i], 'top25_count'] = (valid_positions <= 25).sum()
                    features.loc[features.index[mask][i], 'missed_cut_count'] = (valid_positions > 65).sum()
                    
                    # Calculate consistency rating
                    total_tournaments = len(valid_positions)
                    if total_tournaments > 0:
                        features.loc[features.index[mask][i], 'top10_rate'] = features.loc[features.index[mask][i], 'top10_count'] / total_tournaments
                        features.loc[features.index[mask][i], 'top25_rate'] = features.loc[features.index[mask][i], 'top25_count'] / total_tournaments
                        features.loc[features.index[mask][i], 'missed_cut_rate'] = features.loc[features.index[mask][i], 'missed_cut_count'] / total_tournaments
                        
                        # Create consistency score (0-100)
                        # 60% weight to position_cv (lower is better), 20% to top10_rate, 20% to missed_cut_rate (negative)
                        if 'position_cv' in features.columns:
                            cv = features.loc[features.index[mask][i], 'position_cv']
                            max_cv = features.loc[mask, 'position_cv'].max()
                            normalized_cv = 1 - (cv / max_cv) if max_cv > 0 else 0
                            
                            top10 = features.loc[features.index[mask][i], 'top10_rate']
                            missed = features.loc[features.index[mask][i], 'missed_cut_rate']
                            
                            consistency_score = (normalized_cv * 0.6) + (top10 * 0.2) + ((1 - missed) * 0.2)
                            features.loc[features.index[mask][i], 'consistency_score'] = consistency_score * 100
    
    return features

def _calculate_momentum_metrics(current_form: pd.DataFrame) -> pd.DataFrame:
    # Start with player_id column
    if 'player_id' not in current_form.columns:
        return pd.DataFrame()
        
    features = current_form[['player_id']].copy()
    
    # Find position columns (last1_position, last2_position, etc.)
    position_cols = [col for col in current_form.columns if col.endswith('_position')]
    position_cols = sorted(position_cols)
    
    if len(position_cols) >= 2:
        # Convert position strings to numeric
        positions = pd.DataFrame()
        for col in position_cols:
            positions[col] = current_form[col].apply(
                lambda x: int(x.replace('T', '')) if isinstance(x, str) and x.startswith('T') 
                else int(x) if pd.notna(x) and str(x).isdigit() 
                else np.nan
            )
        
        # Calculate momentum metrics
        # Most recent tournament vs. average of previous tournaments
        if 'last1_position' in positions.columns:
            mask = positions['last1_position'].notna()
            
            if mask.any():
                for i, player_id in enumerate(features.loc[mask, 'player_id']):
                    player_positions = positions.loc[positions.index[mask][i]]
                    most_recent = player_positions['last1_position']
                    
                    # Get previous tournaments (last2, last3, etc.)
                    previous_cols = [col for col in position_cols if col != 'last1_position']
                    previous_positions = player_positions[previous_cols]
                    valid_previous = previous_positions.dropna()
                    
                    if len(valid_previous) > 0:
                        avg_previous = valid_previous.mean()
                        
                        # Calculate momentum (positive = improvement, negative = decline)
                        momentum = avg_previous - most_recent
                        features.loc[features.index[mask][i], 'position_momentum'] = momentum
                        
                        # Calculate recent trajectory
                        # Looking at last3 tournaments to detect trends
                        recent_cols = ['last1_position', 'last2_position', 'last3_position']
                        recent_cols = [col for col in recent_cols if col in player_positions.index]
                        
                        if len(recent_cols) >= 2:
                            recent_positions = player_positions[recent_cols]
                            valid_recent = recent_positions.dropna()
                            
                            if len(valid_recent) >= 2:
                                # Get indices (0,1,2,...) for valid positions and convert to list
                                indices = np.arange(len(valid_recent)).reshape(-1, 1)
                                position_values = valid_recent.values.reshape(-1, 1)
                                
                                # Calculate slope using numpy's polyfit
                                if len(indices) > 0 and len(position_values) > 0:
                                    slope = np.polyfit(indices.flatten(), position_values.flatten(), 1)[0]
                                    features.loc[features.index[mask][i], 'recent_trajectory'] = -slope  # Negative slope = improvement
    
    # Add recency bias score - how recent form compares to longer-term average
    # This measures if a player is "hot" or "cold" compared to their usual performance
    if 'tournament_history' in current_form.columns:
        # Would need historical tournament data, which is more complex
        pass
    
    return features

def _calculate_sg_trends(current_form: pd.DataFrame) -> pd.DataFrame:
  
    # Start with player_id column
    if 'player_id' not in current_form.columns:
        return pd.DataFrame()
        
    features = current_form[['player_id']].copy()
    
    # Find SG columns
    sg_cols = [col for col in current_form.columns if col.startswith('sg_') and col.endswith('_value')]
    
    if sg_cols:
        # For each SG category, check if most recent value is better/worse than average
        for col in sg_cols:
            sg_category = col.replace('_value', '')
            
            mask = current_form[col].notna()
            if mask.any():
                features.loc[mask, f'{sg_category}'] = current_form.loc[mask, col]
                
                # Add trend indicators if we have multiple tournaments
                # This would require historical SG data across tournaments,
                # which is more complex to determine from current_form alone
    
    return features

def _merge_feature_sets(feature_sets: List[pd.DataFrame], on: str = 'player_id') -> pd.DataFrame:

    if not feature_sets:
        return pd.DataFrame()
    
    # Start with the first feature set
    merged = feature_sets[0].copy()
    
    # Merge with remaining feature sets
    for i, features in enumerate(feature_sets[1:], 1):
        if features.empty or on not in features.columns:
            continue
            
        # Get column suffixes to avoid conflicts
        left_suffix = ''
        right_suffix = f'_{i}'
        
        # Merge current features with accumulated result
        merged = pd.merge(
            merged,
            features,
            on=on,
            how='outer',
            suffixes=(left_suffix, right_suffix)
        )
    
    return merged