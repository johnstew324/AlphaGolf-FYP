import pandas as pd
import numpy as np

def create_recent_performance_features(player_ids, tournament_id, processors, window_size=5):
    if 'current_form' in processors:
        current_form = processors['current_form'].extract_features(player_ids=player_ids, tournament_id=tournament_id)
        if not current_form.empty:
            return _process_recent_form(current_form, window_size)
    return pd.DataFrame({'player_id': player_ids})

def _process_recent_form(current_form, window_size=5):
    if 'player_id' not in current_form.columns:
        return pd.DataFrame()
        
    features = current_form[['player_id']].copy()
    position_cols = [col for col in current_form.columns if col.endswith('_position')]
    position_cols = sorted(position_cols)[:window_size]
    
    if position_cols:
        positions = pd.DataFrame()
        for col in position_cols:
            positions[col] = current_form[col].apply(
                lambda x: int(x.replace('T', '')) if isinstance(x, str) and x.startswith('T') 
                else int(x) if pd.notna(x) and str(x).isdigit() 
                else np.nan
            )
        
        features['valid_tournaments'] = positions.notna().sum(axis=1)
        mask = features['valid_tournaments'] > 0
        
        if mask.any():
            features.loc[mask, 'avg_recent_position'] = positions.loc[mask].mean(axis=1)
            
            if len(position_cols) >= 2:
                for i, player_id in enumerate(features.loc[mask, 'player_id']):
                    player_positions = positions.loc[positions.index[mask][i]]
                    valid_positions = player_positions.dropna()
                    
                    if len(valid_positions) >= 2:
                        indices = np.arange(len(valid_positions)).reshape(-1, 1)
                        position_values = valid_positions.values.reshape(-1, 1)
                        if len(indices) > 0 and len(position_values) > 0:
                            slope = np.polyfit(indices.flatten(), position_values.flatten(), 1)[0]
                            features.loc[features.index[mask][i], 'position_trend'] = slope
    
    if position_cols and len(position_cols) > 0:
        weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(position_cols)]
        weights = [w / sum(weights) for w in weights]
        
        for i, player_id in enumerate(features['player_id']):
            player_positions = positions.loc[i, position_cols].copy()
            valid_mask = player_positions.notna()
            
            if valid_mask.any():
                valid_positions = player_positions[valid_mask].values
                valid_weights = [weights[j] for j, is_valid in enumerate(valid_mask) if is_valid]
                valid_weights = [w / sum(valid_weights) for w in valid_weights]
                weighted_avg = sum(p * w for p, w in zip(valid_positions, valid_weights))
                features.loc[i, 'weighted_recent_position'] = weighted_avg
    
    return features

def _calculate_consistency_metrics(current_form):
    if 'player_id' not in current_form.columns:
        return pd.DataFrame()
        
    features = current_form[['player_id']].copy()
    position_cols = [col for col in current_form.columns if col.endswith('_position')]
    position_cols = sorted(position_cols)
    
    if position_cols:
        positions = pd.DataFrame()
        for col in position_cols:
            positions[col] = current_form[col].apply(
                lambda x: int(x.replace('T', '')) if isinstance(x, str) and x.startswith('T') 
                else int(x) if pd.notna(x) and str(x).isdigit() 
                else np.nan
            )
        
        mask = positions.notna().sum(axis=1) > 1
        
        if mask.any():
            features.loc[mask, 'position_std'] = positions.loc[mask].std(axis=1)
            
            for i, player_id in enumerate(features.loc[mask, 'player_id']):
                player_positions = positions.loc[positions.index[mask][i]]
                valid_positions = player_positions.dropna()
                
                if len(valid_positions) > 0:
                    features.loc[features.index[mask][i], 'top10_count'] = (valid_positions <= 10).sum()
                    features.loc[features.index[mask][i], 'top25_count'] = (valid_positions <= 25).sum()
                    
                    total_tournaments = len(valid_positions)
                    if total_tournaments > 0:
                        features.loc[features.index[mask][i], 'top10_rate'] = features.loc[features.index[mask][i], 'top10_count'] / total_tournaments
    
    return features

def create_momentum_features(player_ids, tournament_id, processors):
    if 'current_form' in processors:
        current_form = processors['current_form'].extract_features(player_ids=player_ids, tournament_id=tournament_id)
        if not current_form.empty:
            return _calculate_momentum_metrics(current_form)
    return pd.DataFrame({'player_id': player_ids})

def _calculate_momentum_metrics(current_form):
    if 'player_id' not in current_form.columns:
        return pd.DataFrame()
        
    features = current_form[['player_id']].copy()
    position_cols = [col for col in current_form.columns if col.endswith('_position')]
    position_cols = sorted(position_cols)
    
    if len(position_cols) >= 2:
        positions = pd.DataFrame()
        for col in position_cols:
            positions[col] = current_form[col].apply(
                lambda x: int(x.replace('T', '')) if isinstance(x, str) and x.startswith('T') 
                else int(x) if pd.notna(x) and str(x).isdigit() 
                else np.nan
            )
        
        if 'last1_position' in positions.columns:
            mask = positions['last1_position'].notna()
            
            if mask.any():
                for i, player_id in enumerate(features.loc[mask, 'player_id']):
                    player_positions = positions.loc[positions.index[mask][i]]
                    most_recent = player_positions['last1_position']
                    
                    previous_cols = [col for col in position_cols if col != 'last1_position']
                    previous_positions = player_positions[previous_cols]
                    valid_previous = previous_positions.dropna()
                    
                    if len(valid_previous) > 0:
                        avg_previous = valid_previous.mean()
                        momentum = avg_previous - most_recent
                        features.loc[features.index[mask][i], 'position_momentum'] = momentum
    
    return features

def create_temporal_features(player_ids, tournament_id, season, processors):
    recent_performance = create_recent_performance_features(player_ids, tournament_id, processors)
    momentum = create_momentum_features(player_ids, tournament_id, processors)
    
    all_features = []
    if not recent_performance.empty and 'player_id' in recent_performance.columns:
        all_features.append(recent_performance)
    
    if not momentum.empty and 'player_id' in momentum.columns:
        all_features.append(momentum)
    
    if not all_features:
        return pd.DataFrame({'player_id': player_ids})
    
    temporal_features = _merge_feature_sets(all_features)
    temporal_features['has_temporal_features'] = 1
    
    return temporal_features

def _merge_feature_sets(feature_sets, on='player_id'):
    if not feature_sets:
        return pd.DataFrame()
    
    merged = feature_sets[0].copy()
    
    for i, features in enumerate(feature_sets[1:], 1):
        if features.empty or on not in features.columns:
            continue
        
        merged = pd.merge(merged, features, on=on, how='outer', suffixes=('', f'_{i}'))
    
    return merged