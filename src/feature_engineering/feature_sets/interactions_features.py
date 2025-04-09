# feature_engineering/feature_sets/interaction_features.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def create_player_course_interactions(player_features: pd.DataFrame, 
                                     course_features: pd.DataFrame) -> pd.DataFrame:
    if 'player_id' not in player_features.columns:
        return pd.DataFrame()
        
    # Extract base player data
    interactions = player_features[['player_id']].copy()
    
    # Check if we have course features
    if course_features.empty:
        return interactions
    
    # Map course features to each player
    for col in course_features.columns:
        if col not in ['player_id', 'tournament_id'] and not col.startswith('has_'):
            # Add course feature as global feature for all players
            interactions[f'course_{col}'] = course_features[col].iloc[0] if len(course_features) > 0 else None
    
    # ======= Create SG × Course Characteristic Interactions =======
    sg_categories = ['sg_ott', 'sg_app', 'sg_atg', 'sg_p', 'sg_tot']
    
    # Key course characteristics that might interact with SG categories
    course_characteristics = {
        'par3_to_par': ['sg_app', 'sg_atg', 'sg_p'],  # Par 3 performance relates to approach, around green, putting
        'par4_to_par': ['sg_ott', 'sg_app'],          # Par 4 performance relates to driving and approach
        'par5_to_par': ['sg_ott', 'sg_app'],          # Par 5 performance relates to driving and approach
        'yards_per_par': ['sg_ott'],                  # Course length interacts with driving
        'scoring_difficulty': sg_categories,          # Overall difficulty interacts with all SG categories
        'bogeys_pct': ['sg_app', 'sg_atg'],           # Bogey avoidance through approach and around green
        'birdies_pct': ['sg_app', 'sg_p'],            # Birdie conversion through approach and putting
        'avg_windspeed': ['sg_ott', 'sg_app']         # Wind affects driving and approach the most
    }
    
    # Calculate interactions for available combinations
    for course_char, sg_list in course_characteristics.items():
        if course_char in course_features.columns:
            course_value = course_features[course_char].iloc[0] if len(course_features) > 0 else None
            
            if course_value is not None and not pd.isna(course_value):
                for sg in sg_list:
                    if sg in player_features.columns:
                        # Create interaction feature
                        interactions[f'{sg}_x_{course_char}'] = (
                            player_features[sg] * course_value
                        )
    
    # ======= Create Par-Type Scoring Advantages =======
    # Example: Player with good par 5 scoring × course with many par 5 scoring opportunities
    par_advantages = {
        'par3_scoring_avg': ['par3_count', 'par3_scoring_avg'],
        'par4_scoring_avg': ['par4_count', 'par4_scoring_avg'],
        'par5_scoring_avg': ['par5_count', 'par5_scoring_avg']
    }
    
    for player_stat, course_stats in par_advantages.items():
        if player_stat in player_features.columns and all(stat in course_features.columns for stat in course_stats):
            # Get course values
            count = course_features[course_stats[0]].iloc[0] if len(course_features) > 0 else None
            avg = course_features[course_stats[1]].iloc[0] if len(course_features) > 0 else None
            
            if count is not None and avg is not None and not pd.isna(count) and not pd.isna(avg):
                # Create advantage score - positive means player is better than course average
                par_type = player_stat.split('_')[0]  # Extract par3, par4, or par5
                interactions[f'{par_type}_advantage'] = avg - player_features[player_stat]
                interactions[f'{par_type}_weighted_advantage'] = (avg - player_features[player_stat]) * count
    
    # ======= Create Putting × Green Interactions =======
    if 'sg_p' in player_features.columns and 'overview_green' in course_features.columns:
        # Create putting surface interaction
        green_type = course_features['overview_green'].iloc[0] if len(course_features) > 0 else None
        
        if green_type is not None and not pd.isna(green_type):
            # Create categorical green type variable
            green_types = ['Bentgrass', 'Bermudagrass', 'Poa annua', 'Paspalum']
            for g_type in green_types:
                if g_type.lower() in str(green_type).lower():
                    interactions[f'sg_p_x_{g_type.lower()}'] = player_features['sg_p']
    
    return interactions

def create_weather_adaptation_features(player_features: pd.DataFrame, 
                                     weather_features: pd.DataFrame) -> pd.DataFrame:
    
    if 'player_id' not in player_features.columns:
        return pd.DataFrame()
        
    # Extract base player data
    interactions = player_features[['player_id']].copy()
    sg_categories = ['sg_ott', 'sg_app', 'sg_atg', 'sg_p', 'sg_tot']
    
    # Check if we have weather features and weather data is available
    if weather_features.empty or ('has_weather_data' in weather_features.columns and 
                                  weather_features['has_weather_data'].iloc[0] == 0):
        return interactions
    
    # ======= Create Weather Condition Interactions =======
    # Weather conditions that might affect player performance
    weather_conditions = {
        'avg_windspeed': ['sg_ott', 'sg_app'],  # Wind affects driving and approach
        'avg_temp': ['sg_p'],                    # Temperature affects putting
        'avg_humidity': ['sg_p'],                # Humidity affects putting
        'total_precip': ['sg_atg', 'sg_p'],      # Precipitation affects short game
        'weather_difficulty': sg_categories      # Overall weather difficulty affects all aspects
    }
    
    # Calculate interactions for available combinations
    for weather_cond, sg_list in weather_conditions.items():
        if weather_cond in weather_features.columns:
            weather_value = weather_features[weather_cond].iloc[0] if len(weather_features) > 0 else None
            
            if weather_value is not None and not pd.isna(weather_value):
                for sg in sg_list:
                    if sg in player_features.columns:
                        # Create interaction feature
                        interactions[f'{sg}_x_{weather_cond}'] = (
                            player_features[sg] * weather_value
                        )
    
    # ======= Create Wind Adaptation Features =======
    if 'avg_windspeed' in weather_features.columns:
        wind_speed = weather_features['avg_windspeed'].iloc[0] if len(weather_features) > 0 else None
        
        if wind_speed is not None and not pd.isna(wind_speed):
            # Classify wind conditions
            if wind_speed < 8:
                wind_condition = 'light'
            elif wind_speed < 15:
                wind_condition = 'moderate'
            else:
                wind_condition = 'strong'
                
            # Create wind condition specific features
            for sg in ['sg_ott', 'sg_app']:
                if sg in player_features.columns:
                    interactions[f'{sg}_in_{wind_condition}_wind'] = player_features[sg]
            
            # Create wind adaptation metric if we have consistency metrics
            if 'position_std' in player_features.columns and 'avg_finish' in player_features.columns:
                # Lower standard deviation means more consistent, better in wind
                interactions['wind_consistency_score'] = (
                    (100 - player_features['position_std'] * 5) * (wind_speed / 15)
                )
    
    return interactions

def create_form_history_interactions(current_form: pd.DataFrame, 
                                   tournament_history: pd.DataFrame) -> pd.DataFrame:
    # Start with player_id column
    if 'player_id' not in current_form.columns:
        return pd.DataFrame()
        
    # Extract base player data
    interactions = current_form[['player_id']].copy()
    
    # Check if we have tournament history features
    if tournament_history.empty:
        return interactions
    
    # Merge data temporarily to calculate interactions
    merged = pd.merge(
        current_form,
        tournament_history,
        on='player_id',
        how='left',
        suffixes=('_form', '_history')
    )
    
    # ======= Create Form × History Interactions =======
    
    # Recent form vs. course history comparison
    if 'recent_avg_finish' in merged.columns and 'avg_finish' in merged.columns:
        # Positive means current form is better than historical average at this course
        merged['form_vs_history_diff'] = merged['avg_finish'] - merged['recent_avg_finish']
        
        # Weighted difference (more weight to strong differences)
        merged['form_vs_history_advantage'] = merged['form_vs_history_diff'] * (
            5 / (merged['avg_finish'] + 5)  # Normalize by historical performance level
        )
    
    # Combine recent strokes gained with historical performance
    sg_categories = ['sg_ott', 'sg_app', 'sg_atg', 'sg_p', 'sg_tot']
    history_metrics = ['avg_finish', 'avg_score_to_par', 'cuts_made_pct']
    
    for sg in sg_categories:
        if sg in merged.columns:
            for hist in history_metrics:
                if hist in merged.columns:
                    # Create interaction name
                    col_name = f'{sg}_x_{hist}'
                    
                    # For avg_finish and avg_score_to_par, lower is better
                    if hist in ['avg_finish', 'avg_score_to_par']:
                        # Higher SG and lower average score/finish is best combination
                        # We invert the historical metric so that positive values are better
                        if hist == 'avg_finish':
                            # Normalize finish position (empirical constants based on typical values)
                            norm_hist = 80 - merged[hist]
                        else:
                            # Score to par is already in reasonable range
                            norm_hist = -1 * merged[hist]
                            
                        # Multiply normalized values
                        merged[col_name] = merged[sg] * norm_hist
                    else:
                        # For cuts_made_pct, higher is better
                        merged[col_name] = merged[sg] * merged[hist]
    
    # Extract only the interaction columns we created
    interaction_cols = ['player_id']
    interaction_cols.extend([col for col in merged.columns if col not in current_form.columns 
                            and col not in tournament_history.columns
                            or col == 'form_vs_history_diff' 
                            or col == 'form_vs_history_advantage'])
    
    interactions = merged[interaction_cols].copy()
    
    return interactions

def create_player_field_strength_features(player_features: pd.DataFrame) -> pd.DataFrame:
    # Start with player_id column
    if 'player_id' not in player_features.columns:
        return pd.DataFrame()
        
    # Extract base player data
    interactions = player_features[['player_id']].copy()
    
    # ======= Create Field Strength Metrics =======
    
    # Get key performance metrics
    perf_metrics = ['sg_tot', 'recent_avg_finish', 'career_success_score']
    
    for metric in perf_metrics:
        if metric in player_features.columns:
            # Calculate field percentile for this metric
            if metric in ['recent_avg_finish']:  # Lower is better
                interactions[f'{metric}_percentile'] = player_features[metric].rank(pct=True, ascending=True)
            else:  # Higher is better
                interactions[f'{metric}_percentile'] = player_features[metric].rank(pct=True, ascending=False)
    
    # Enhanced field strength with OWGR
    if 'owgr' in player_features.columns:
        # Calculate field strength percentile based on OWGR
        interactions['owgr_percentile'] = player_features['owgr'].rank(pct=True, ascending=True)
        
        # Add to list of percentile columns
        percentile_cols = [col for col in interactions.columns if col.endswith('_percentile')]
        percentile_cols.append('owgr_percentile')
        
        # Add raw OWGR-based field strength
        interactions['owgr_field_strength'] = (
            100 - player_features['owgr'].rank(ascending=True) * 100 / len(player_features)
        )
        
        # Field strength vs OWGR expectation
        if 'field_strength_score' in interactions.columns:
            # Positive means player is performing better than their OWGR would predict
            interactions['field_vs_owgr'] = (
                interactions['field_strength_score'] - interactions['owgr_field_strength']
            )
    else:
        # Calculate percentile columns if OWGR not available
        percentile_cols = [col for col in interactions.columns if col.endswith('_percentile')]
    
    # Calculate composite field strength score
    if percentile_cols:
        interactions['field_strength_score'] = interactions[percentile_cols].mean(axis=1) * 100
    
    return interactions

def create_meta_features(player_features: pd.DataFrame, 
                       course_features: pd.DataFrame,
                       tournament_history: pd.DataFrame,
                       weather_features: pd.DataFrame) -> pd.DataFrame:
    # Start with player_id column
    if 'player_id' not in player_features.columns:
        return pd.DataFrame()
        
    # Extract base player data
    meta_features = player_features[['player_id']].copy()
    
    # ======= Create Tournament Success Likelihood Score =======
    success_components = []
    
    # Component 1: Course history (if available)
    if not tournament_history.empty and 'course_history_score' in tournament_history.columns:
        meta_features['history_component'] = tournament_history['course_history_score']
        success_components.append('history_component')
    
    # Component 2: Current form (if available)
    if 'recent_form_score' in player_features.columns:
        meta_features['form_component'] = player_features['recent_form_score']
        success_components.append('form_component')
    elif 'sg_tot' in player_features.columns:
        # Normalize SG: Total to 0-100 scale as alternative form metric
        sg_tot = player_features['sg_tot']
        meta_features['form_component'] = (sg_tot - sg_tot.min()) / (sg_tot.max() - sg_tot.min()) * 100
        success_components.append('form_component')
    
    # Component 3: Course fit (if available)
    if 'course_fit_score' in player_features.columns:
        meta_features['fit_component'] = player_features['course_fit_score']
        success_components.append('fit_component')
    else:
        # Create basic course fit from SG categories and course characteristics
        if not course_features.empty and 'par3_to_par' in course_features.columns:
            # Simple course fit metric based on par 3/4/5 scoring
            par3_value = course_features['par3_to_par'].iloc[0] if len(course_features) > 0 else 0
            par4_value = course_features['par4_to_par'].iloc[0] if len(course_features) > 0 else 0
            par5_value = course_features['par5_to_par'].iloc[0] if len(course_features) > 0 else 0
            
            # Create a Series with the same index as player_features for the fit scores
            fit_score = pd.Series(50, index=player_features.index)  # Start at neutral
            
            if 'sg_app' in player_features.columns and abs(par3_value) > 0.05:
                # Par 3 performance correlates with approach play
                fit_score += player_features['sg_app'] * (20 if par3_value < 0 else -20)
            
            if 'sg_ott' in player_features.columns and abs(par5_value) > 0.05:
                # Par 5 performance correlates with driving
                fit_score += player_features['sg_ott'] * (20 if par5_value < 0 else -20)
            
            meta_features['fit_component'] = fit_score.clip(0, 100)
            success_components.append('fit_component')
    
    
    # Component 4: Weather adaptation (if available)
    if not weather_features.empty and 'weather_difficulty' in weather_features.columns:
        difficulty = weather_features['weather_difficulty'].iloc[0] if len(weather_features) > 0 else None
        
        if difficulty is not None and not pd.isna(difficulty) and difficulty > 0:
            # Create weather adaptation score
            if 'position_std' in player_features.columns:
                # More consistent players do better in difficult weather
                consistency = 100 - player_features['position_std'] * 5
                meta_features['weather_component'] = consistency * (difficulty / 10)
                success_components.append('weather_component')
    
    # Calculate weighted tournament success likelihood
    if success_components:
        # Define component weights (adjust based on importance)
        weights = {
            'form_component': 0.40,      # Current form is most important
            'history_component': 0.25,   # Course history is next
            'fit_component': 0.25,       # Course fit is also important
            'weather_component': 0.10    # Weather adaptation is a minor factor
        }
        
        # Calculate weighted sum, adjusting for missing components
        weighted_sum = 0
        total_weight = 0
        
        for component in success_components:
            if component in weights:
                weighted_sum += meta_features[component] * weights[component]
                total_weight += weights[component]
        
        # Normalize by actual weight used
        if total_weight > 0:
            meta_features['tournament_success_likelihood'] = weighted_sum / total_weight
    
    # ======= Create Victory Potential Score =======
    victory_components = []
    
    # Component 1: Field strength position (if available)
    if 'field_strength_score' in player_features.columns:
        meta_features['field_position'] = player_features['field_strength_score']
        victory_components.append('field_position')
    
    # Component 2: Win history (if available)
    if 'career_win_rate' in player_features.columns:
        meta_features['win_history'] = player_features['career_win_rate'] * 100
        victory_components.append('win_history')
    
    # Component 3: Course win/top finish history (if available)
    if not tournament_history.empty:
        if 'wins_1' in tournament_history.columns:
            meta_features['course_win_history'] = tournament_history['wins_1'] * 100
            victory_components.append('course_win_history')
        elif 'top_10_finishes' in tournament_history.columns:
            meta_features['course_win_history'] = tournament_history['top_10_finishes'] * 20
            victory_components.append('course_win_history')
    
    # Component 4: OWGR position (if available)
    if 'owgr_score_norm' in player_features.columns:
        meta_features['owgr_component'] = player_features['owgr_score_norm']
        victory_components.append('owgr_component')
    elif 'owgr' in player_features.columns:
        # Simple inversion with normalization
        owgr = player_features['owgr']
        meta_features['owgr_component'] = (200 - owgr.clip(1, 200)) / 2
        victory_components.append('owgr_component')
    
    # Calculate victory potential score with updated weights
    if victory_components and 'tournament_success_likelihood' in meta_features.columns:
        # OWGR is a strong predictor of victory potential
        meta_features['victory_potential'] = (
            meta_features[victory_components].mean(axis=1) * 0.6 +
            meta_features['tournament_success_likelihood'] * 0.4
        )
    elif victory_components:
        # Calculate based only on victory components if no tournament success score
        meta_features['victory_potential'] = meta_features[victory_components].mean(axis=1)
    
    return meta_features

def create_interaction_features(player_ids: List[str], tournament_id: str, 
                              season: int, processors: Dict, 
                              base_features: pd.DataFrame,
                              temporal_features: pd.DataFrame) -> pd.DataFrame:
    # If base_features is empty, we can't create interactions
    if base_features.empty:
        return pd.DataFrame({'player_id': player_ids})
    
    # Extract relevant feature sets from base_features
    # We'll separate player-level features from tournament/course-level features
    
    # Player performance features
    player_features = base_features.copy()
    
    # Get course features
    if 'course_stats' in processors:
        course_features = processors['course_stats'].extract_features(
            tournament_id=tournament_id,
            season=season
        )
    else:
        course_features = pd.DataFrame()
    
    # Get weather features
    if 'tournament_weather' in processors:
        weather_features = processors['tournament_weather'].extract_features(
            tournament_ids=tournament_id,
            season=season
        )
    else:
        weather_features = pd.DataFrame()
    
    # Get tournament history
    if 'tournament_history' in processors:
        # Handle the special tournament ID format
        if tournament_id.startswith("R") and len(tournament_id) >= 8:
            special_id = "R2025" + tournament_id[5:]
        else:
            special_id = tournament_id
            
        tournament_history = processors['tournament_history'].extract_features(
            tournament_id=special_id,
            player_ids=player_ids
        )
    else:
        tournament_history = pd.DataFrame()
    
    # Merge base and temporal features if available
    if not temporal_features.empty and 'player_id' in temporal_features.columns:
        player_features = pd.merge(
            player_features,
            temporal_features,
            on='player_id',
            how='left'
        )
    
    # Generate individual feature sets
    player_course = create_player_course_interactions(player_features, course_features)
    weather_adaptation = create_weather_adaptation_features(player_features, weather_features)
    form_history = create_form_history_interactions(player_features, tournament_history)
    field_strength = create_player_field_strength_features(player_features)
    meta = create_meta_features(player_features, course_features, tournament_history, weather_features)
    
    # Combine all features
    all_features = []
    
    # Start with player IDs if all features are empty
    if (player_course.empty and weather_adaptation.empty and 
        form_history.empty and field_strength.empty and meta.empty):
        return pd.DataFrame({'player_id': player_ids})
    
    # Add features that have data
    if not player_course.empty:
        all_features.append(player_course)
    
    if not weather_adaptation.empty:
        all_features.append(weather_adaptation)
    
    if not form_history.empty:
        all_features.append(form_history)
    
    if not field_strength.empty:
        all_features.append(field_strength)
    
    if not meta.empty:
        all_features.append(meta)
    
    # Merge all feature sets
    interaction_features = _merge_feature_sets(all_features, on='player_id')
    
    # Add availability metadata
    interaction_features['has_interaction_features'] = 1
    
    return interaction_features

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