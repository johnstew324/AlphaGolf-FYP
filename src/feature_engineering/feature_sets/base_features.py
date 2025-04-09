# feature_engineering/feature_sets/base_features.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def create_player_performance_features(player_ids: List[str], season: int, 
                                      processors: Dict, tournament_id: str = None) -> pd.DataFrame:
    """
    Create core player performance features from player statistics.
    
    Args:
        player_ids: List of player IDs to include
        season: The current season
        processors: Dictionary of processor instances
        tournament_id: Optional tournament ID
        
    Returns:
        DataFrame with player performance features
    """
    # Start with basic player stats
    player_stats = processors['player_form'].extract_features(
        player_ids=player_ids, 
        season=season, 
        tournament_id=tournament_id
    )
    
    # Create base features DataFrame
    if player_stats.empty:
        return pd.DataFrame()
        
    # Extract core feature columns
    features = player_stats.copy()
    
    # Add career and profile data if available
    if 'player_profile' in processors:
        profile_features = processors['player_profile'].extract_features(
            player_ids=player_ids,
            season=season
        )
        
        owgr_features = processors['player_profile']._process_owgr_data(profile_features)
        
        if not owgr_features.empty:
            features = pd.merge(
                features,
                owgr_features,
                on='player_id',
                how='left'
            )
        
        if not profile_features.empty:
            features = pd.merge(
                features,
                profile_features,
                on='player_id',
                how='left',
                suffixes=('', '_profile')
            )
    
    # Add availability flags
    features['has_performance_data'] = 1
    
    # Add current form data if available (2024+)
    if season >= 2024 and 'current_form' in processors:
        current_form = processors['current_form'].extract_features(
            player_ids=player_ids,
            tournament_id=tournament_id
        )
        
        if not current_form.empty:
            # Merge current form features
            form_cols = [col for col in current_form.columns if col != 'player_id']
            current_form_subset = current_form[['player_id'] + form_cols].copy()
            
            features = pd.merge(
                features,
                current_form_subset,
                on='player_id',
                how='left',
                suffixes=('', '_current')
            )
            
            features['has_current_form'] = 1
        else:
            features['has_current_form'] = 0
    else:
        features['has_current_form'] = 0
    
    # Calculate additional derived metrics
    features = _add_derived_performance_metrics(features)
    
    return features

def create_tournament_history_features(tournament_id: str, player_ids: List[str], 
                                     processors: Dict, season: int = None) -> pd.DataFrame:

    if tournament_id.startswith("R") and len(tournament_id) >= 8:
        special_id = "R2025" + tournament_id[5:]
    else:
        special_id = tournament_id
    
    # Get player tournament history
    tournament_history = processors['tournament_history'].extract_features(
        tournament_id=special_id,
        player_ids=player_ids,
        season=season
    )
    
    if tournament_history.empty:
        return pd.DataFrame()
    
    # Create base features
    features = tournament_history.copy()
    features['has_tournament_history'] = 1
    
    # Add tournament history stats if available (2024+)
    if season and season >= 2024 and 'tournament_history_stats' in processors:
        history_stats = processors['tournament_history_stats'].extract_features(
            tournament_id=tournament_id,
            player_ids=player_ids
        )
        
        if not history_stats.empty:
            # Filter columns to avoid duplicates
            stats_cols = [col for col in history_stats.columns 
                         if col not in ['player_id', 'tournament_id', 'total_rounds']]
            
            history_stats_subset = history_stats[['player_id'] + stats_cols].copy()
            
            features = pd.merge(
                features,
                history_stats_subset,
                on='player_id',
                how='left',
                suffixes=('', '_stats')
            )
            
            features['has_history_stats'] = 1
        else:
            features['has_history_stats'] = 0
    else:
        features['has_history_stats'] = 0
    
    # Add derived metrics
    features = _add_derived_history_metrics(features)
    
    return features

def create_course_features(tournament_id: str, player_ids: List[str], 
                          processors: Dict, season: int = None) -> pd.DataFrame:
    if not season and tournament_id.startswith("R") and len(tournament_id) >= 8:
        try:
            season = int(tournament_id[1:5])
        except ValueError:
            pass
    
    # Extract course stats
    course_features = None
    
    # Base course data is available for all tournaments from tournament_history
    if 'tournament_history' in processors:
        # Handle the special tournament ID format
        if tournament_id.startswith("R") and len(tournament_id) >= 8:
            special_id = "R2025" + tournament_id[5:]
        else:
            special_id = tournament_id
            
        # Extract tournament-level data
        tournament_data = processors['tournament_history'].extract_features(
            tournament_id=special_id,
            player_ids=None,  # No player filtering for course data
            season=season
        )
        
        if not tournament_data.empty:
            course_features = tournament_data.copy()
    
    # Add detailed course stats if available (2023+)
    has_course_stats = False
    if season and season >= 2023 and 'course_stats' in processors:
        course_stats = processors['course_stats'].extract_features(
            tournament_id=tournament_id,
            season=season
        )
        
        if not course_stats.empty:
            has_course_stats = True
            
            if course_features is not None:
                # Merge with existing features
                # First ensure we have a common key for merging
                if 'tournament_id' in course_stats.columns:
                    # Filter columns to avoid duplicates
                    stats_cols = [col for col in course_stats.columns if col != 'tournament_id']
                    course_stats_subset = course_stats[['tournament_id'] + stats_cols].copy()
                    
                    course_features = pd.merge(
                        course_features,
                        course_stats_subset,
                        on='tournament_id',
                        how='left',
                        suffixes=('', '_stats')
                    )
            else:
                course_features = course_stats.copy()
    
    # Return early if no course features found
    if course_features is None or course_features.empty:
        return pd.DataFrame()
    
    # Add availability flag
    course_features['has_course_stats'] = 1 if has_course_stats else 0
    
    # Create player-specific version if player_ids provided
    if player_ids:
        # Create a player-level dataframe
        player_df = pd.DataFrame({'player_id': player_ids})
        
        # Add tournament_id to each player
        player_df['tournament_id'] = tournament_id
        
        # Get course-fit data if available (2025+)
        if season and season >= 2025 and 'course_fit' in processors:
            course_fit = processors['course_fit'].extract_features(
                tournament_id=tournament_id,
                player_ids=player_ids
            )
            
            if not course_fit.empty:
                # Merge course fit with player dataframe
                player_df = pd.merge(
                    player_df,
                    course_fit,
                    on=['player_id', 'tournament_id'],
                    how='left'
                )
                
                player_df['has_course_fit'] = 1
            else:
                player_df['has_course_fit'] = 0
        else:
            player_df['has_course_fit'] = 0
        
        # Now merge course features with player dataframe
        # First ensure course_features has tournament_id column
        if 'tournament_id' in course_features.columns:
            # Keep only columns that aren't player-specific
            course_cols = [col for col in course_features.columns 
                          if col != 'player_id' and not col.startswith('player_')]
            
            # Merge course data into player dataframe
            player_course_features = pd.merge(
                player_df,
                course_features[course_cols],
                on='tournament_id',
                how='left'
            )
            
            return player_course_features
        else:
            # If no tournament_id column, add it and return
            player_df['tournament_id'] = tournament_id
            return player_df
    else:
        # If no player_ids, just return course features
        return course_features

def create_weather_features(tournament_id: str, player_ids: List[str], 
                           processors: Dict, season: int = None) -> pd.DataFrame:
    if not season and tournament_id.startswith("R") and len(tournament_id) >= 8:
        try:
            season = int(tournament_id[1:5])
        except ValueError:
            pass
    

    if season and season < 2022:
        # Create empty DataFrame with tournament_id and has_weather_data=0
        weather_features = pd.DataFrame({'tournament_id': [tournament_id], 'has_weather_data': [0]})
    elif 'tournament_weather' in processors:
        # Extract weather data
        weather_features = processors['tournament_weather'].extract_features(
            tournament_ids=tournament_id,
            season=season
        )
        
        if weather_features.empty:
            # Create empty DataFrame with tournament_id and has_weather_data=0
            weather_features = pd.DataFrame({'tournament_id': [tournament_id], 'has_weather_data': [0]})
    else:
        # Create empty DataFrame with tournament_id and has_weather_data=0
        weather_features = pd.DataFrame({'tournament_id': [tournament_id], 'has_weather_data': [0]})
    
    # Create player-specific version if player_ids provided
    if player_ids:
        # Create a player-level dataframe
        player_df = pd.DataFrame({'player_id': player_ids})
        
        # Add tournament_id to each player
        player_df['tournament_id'] = tournament_id
        
        # Now merge weather features with player dataframe
        # First ensure weather_features has tournament_id column
        if 'tournament_id' in weather_features.columns:
            # Keep only columns that aren't player-specific
            weather_cols = [col for col in weather_features.columns 
                          if col != 'player_id' and not col.startswith('player_')]
            
            # Merge weather data into player dataframe
            player_weather_features = pd.merge(
                player_df,
                weather_features[weather_cols],
                on='tournament_id',
                how='left'
            )
            
            return player_weather_features
        else:
            # If no tournament_id column, add it and return
            player_df['tournament_id'] = tournament_id
            player_df['has_weather_data'] = 0
            return player_df
    else:
        # If no player_ids, just return weather features
        return weather_features

def create_career_context_features(player_ids: List[str], season: int, 
                                 processors: Dict) -> pd.DataFrame:
    if 'player_career' in processors:
        career_features = processors['player_career'].extract_features(
            player_ids=player_ids,
            season=season
        )
        
        if not career_features.empty:
            career_features['has_career_data'] = 1
            
            # Add derived metrics
            career_features = _add_derived_career_metrics(career_features)
            
            return career_features
    return pd.DataFrame({'player_id': player_ids, 'has_career_data': 0})

def create_scorecard_features(tournament_id: str, player_ids: List[str], 
                            processors: Dict, season: int = None) -> pd.DataFrame:
    if 'scorecard' in processors:
        scorecard_features = processors['scorecard'].extract_features(
            tournament_id=tournament_id,
            player_ids=player_ids,
            season=season
        )
        
        if not scorecard_features.empty:
            scorecard_features['has_scorecard_data'] = 1
            return scorecard_features
    
    # If no data or processor, return empty DataFrame with player_ids
    return pd.DataFrame({'player_id': player_ids, 'tournament_id': tournament_id, 'has_scorecard_data': 0})




def create_base_features(tournament_id: str, season: int, player_ids: List[str], 
                        processors: Dict, data_extractor=None) -> pd.DataFrame:
    performance_features = create_player_performance_features(player_ids, season, processors, tournament_id)
    history_features = create_tournament_history_features(tournament_id, player_ids, processors, season)
    career_features = create_career_context_features(player_ids, season, processors)
    course_features = create_course_features(tournament_id, player_ids, processors, season)
    weather_features = create_weather_features(tournament_id, player_ids, processors, season)
    scorecard_features = create_scorecard_features(tournament_id, player_ids, processors, season)
    
    # Add position data if data_extractor is provided
    position_features = pd.DataFrame()
    if data_extractor is not None:
        history_tournament_id = tournament_id
        if tournament_id.startswith("R") and len(tournament_id) >= 8:
            tournament_part = tournament_id[5:]
            history_tournament_id = f"R2025{tournament_part}"
        
        position_features = extract_position_and_winner_data(
            data_extractor,
            history_tournament_id,
            player_ids
        )
    
    # Combine all features
    all_features = []
    
    # Start with player IDs if all features are empty
    if (performance_features.empty and history_features.empty and career_features.empty and 
        course_features.empty and weather_features.empty and scorecard_features.empty and
        position_features.empty):
        return pd.DataFrame({'player_id': player_ids, 'tournament_id': tournament_id})
    
    # Add features that have data
    if not performance_features.empty:
        all_features.append(performance_features)
    
    if not history_features.empty:
        all_features.append(history_features)
    
    if not career_features.empty:
        all_features.append(career_features)
    
    if not course_features.empty:
        all_features.append(course_features)
    
    if not weather_features.empty:
        all_features.append(weather_features)
    
    if not scorecard_features.empty:
        all_features.append(scorecard_features)
    
    if not position_features.empty:
        all_features.append(position_features)
        has_position = 1
    else:
        has_position = 0
    
    # Merge all feature sets
    base_features = _merge_feature_sets(all_features, on='player_id')
    
    # Add metadata
    if not base_features.empty:
        base_features['feature_year'] = season
        base_features['tournament_id_standard'] = tournament_id
        base_features['has_position_data'] = has_position
        
        # Calculate data completeness
        has_columns = [col for col in base_features.columns if col.startswith('has_')]
        if has_columns:
            base_features['data_completeness'] = base_features[has_columns].sum(axis=1) / len(has_columns)
    
    return base_features

def _add_derived_performance_metrics(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    
    # SG category ratios (if available)
    sg_categories = ['sg_ott', 'sg_app', 'sg_atg', 'sg_p', 'sg_tot']
    
    if all(cat in df.columns for cat in sg_categories):
        # Calculate SG category contributions
        if df['sg_tot'].abs().max() > 0:
            for cat in sg_categories[:-1]:  # Exclude sg_tot
                df[f'{cat}_contribution'] = df[cat] / df['sg_tot'].where(df['sg_tot'] != 0, np.nan)
        
        # Long game vs. short game ratio
        df['long_game'] = df['sg_ott'] + df['sg_app']
        df['short_game'] = df['sg_atg'] + df['sg_p']
        
        # Calculate balance between long and short games
        df['long_short_ratio'] = df['long_game'] / df['short_game'].where(df['short_game'] != 0, np.nan)
        
        # Calculate primary strength (highest SG category)
        strength_cols = ['sg_ott', 'sg_app', 'sg_atg', 'sg_p']
        for i, row in df.iterrows():
            values = [row[col] if pd.notna(row[col]) else -999 for col in strength_cols]
            if max(values) > -999:
                max_idx = values.index(max(values))
                df.loc[i, 'primary_strength'] = strength_cols[max_idx]
    
    # Recent form metrics (if available)
    form_cols = [col for col in df.columns if 'recent_' in col]
    if form_cols:
        # Check for specific metrics
        if 'recent_top10_rate' in df.columns and 'recent_avg_finish' in df.columns:
            # Create a combined form score (better finish + higher top 10 rate = higher score)
            max_finish = df['recent_avg_finish'].max()
            if max_finish > 0:
                finish_score = 1 - (df['recent_avg_finish'] / max_finish)
                df['recent_form_score'] = (finish_score * 0.5) + (df['recent_top10_rate'] * 0.5)
                
                # Add categorical form rating
                conditions = [
                    (df['recent_form_score'] >= 0.8),
                    (df['recent_form_score'] >= 0.6) & (df['recent_form_score'] < 0.8),
                    (df['recent_form_score'] >= 0.4) & (df['recent_form_score'] < 0.6),
                    (df['recent_form_score'] >= 0.2) & (df['recent_form_score'] < 0.4),
                    (df['recent_form_score'] < 0.2)
                ]
                
                choices = ['Excellent', 'Good', 'Average', 'Poor', 'Very Poor']
                
                df['recent_form_rating'] = np.select(conditions, choices, default='Unknown')
    
    return df

def _add_derived_history_metrics(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    
    # Calculate additional history metrics if we have basic data
    if 'appearances' in df.columns:
        # Calculate consistency ratio (top 25 finishes / appearances)
        if 'top_25_finishes' in df.columns:
            df['consistency_ratio'] = df['top_25_finishes'] / df['appearances']
        
        # Calculate success ratio (top 10 finishes / appearances)
        if 'top_10_finishes' in df.columns:
            df['success_ratio'] = df['top_10_finishes'] / df['appearances']
        
        # Create experience level categories
        df['tournament_experience'] = pd.cut(
            df['appearances'],
            bins=[0, 1, 3, 5, float('inf')],
            labels=['First Time', 'Limited', 'Moderate', 'Extensive']
        )
    
    # Calculate course history score (0-100)
    if all(col in df.columns for col in ['avg_finish', 'best_finish', 'cuts_made_pct']):
        # Normalize average finish (lower is better)
        max_finish = df['avg_finish'].max()
        if max_finish > 0:
            norm_avg_finish = 1 - (df['avg_finish'] / max_finish)
            
            # Normalize best finish (lower is better)
            max_best = df['best_finish'].max()
            if max_best > 0:
                norm_best_finish = 1 - (df['best_finish'] / max_best)
                
                # Calculate history score (weighted average of components)
                df['course_history_score'] = (
                    norm_avg_finish * 0.4 +  # 40% weight on average finish
                    norm_best_finish * 0.3 +  # 30% weight on best finish
                    df['cuts_made_pct'] * 0.3  # 30% weight on cuts made percentage
                ) * 100  # Scale to 0-100
                
                # Add categorical history rating
                conditions = [
                    (df['course_history_score'] >= 80),
                    (df['course_history_score'] >= 60) & (df['course_history_score'] < 80),
                    (df['course_history_score'] >= 40) & (df['course_history_score'] < 60),
                    (df['course_history_score'] >= 20) & (df['course_history_score'] < 40),
                    (df['course_history_score'] < 20)
                ]
                
                choices = ['Excellent', 'Good', 'Average', 'Poor', 'Very Poor']
                
                df['course_history_rating'] = np.select(conditions, choices, default='Unknown')
    
    return df

def _add_derived_career_metrics(features: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived career metrics to the feature set.
    
    Args:
        features: DataFrame with raw career metrics
        
    Returns:
        DataFrame with added derived metrics
    """
    df = features.copy()
    
    # Calculate additional career metrics
    if 'career_span' in df.columns:
        # Create experience categories
        df['experience_level'] = pd.cut(
            df['career_span'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['Rookie', 'Early Career', 'Mid Career', 'Veteran']
        )
    
    # Calculate win rate if we have events and wins
    if 'career_events' in df.columns and 'career_wins' in df.columns:
        df['career_win_rate'] = df['career_wins'] / df['career_events']
    
    # Calculate cuts made rate
    if 'career_cuts_made' in df.columns and 'career_events' in df.columns:
        df['career_cuts_made_rate'] = df['career_cuts_made'] / df['career_events']
    
    # Calculate top 10 rate
    if 'career_top10' in df.columns and 'career_events' in df.columns:
        df['career_top10_rate'] = df['career_top10'] / df['career_events']
    
    # Create career success score (0-100)
    if all(col in df.columns for col in ['career_win_rate', 'career_top10_rate', 'career_cuts_made_rate']):
        df['career_success_score'] = (
            df['career_win_rate'] * 50 +  # 50% weight on win rate
            df['career_top10_rate'] * 30 +  # 30% weight on top 10 rate
            df['career_cuts_made_rate'] * 20  # 20% weight on cuts made rate
        ) * 100  # Scale to 0-100
        
        # Add career success rating
        conditions = [
            (df['career_success_score'] >= 40),
            (df['career_success_score'] >= 30) & (df['career_success_score'] < 40),
            (df['career_success_score'] >= 20) & (df['career_success_score'] < 30),
            (df['career_success_score'] >= 10) & (df['career_success_score'] < 20),
            (df['career_success_score'] < 10)
        ]
        
        choices = ['Elite', 'Very Good', 'Good', 'Average', 'Below Average']
        
        df['career_success_rating'] = np.select(conditions, choices, default='Unknown')
    
    return df

def _merge_feature_sets(feature_sets: List[pd.DataFrame], on: str = 'player_id') -> pd.DataFrame:
    """
    Merge multiple feature sets intelligently, handling duplicates and conflicts.
    
    Args:
        feature_sets: List of DataFrames to merge
        on: Key column for merging
        
    Returns:
        Merged DataFrame
    """
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
        
        # Resolve duplicate tournament_id columns
        tid_cols = [col for col in merged.columns if col.startswith('tournament_id') and col != 'tournament_id']
        if tid_cols and 'tournament_id' in merged.columns:
            for col in tid_cols:
                # Fill NAs in main tournament_id with values from duplicate columns
                merged['tournament_id'] = merged['tournament_id'].fillna(merged[col])
                # Drop the duplicate column
                merged = merged.drop(columns=[col])
    
    return merged