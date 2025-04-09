# feature_engineering/feature_selection/feature_transformer.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import List, Dict, Tuple, Union, Optional
import re
from datetime import datetime

class FeatureTransformer:
    
    def __init__(self, drop_timestamps=True, special_player_ids=None):
        self.drop_timestamps = drop_timestamps
        self.special_player_ids = special_player_ids or []
        self.transformers = {}
        self.feature_types = {}
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.feature_groups = {}
        
    def fit_transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df = features_df.copy()
        
        # Analyze feature types
        self._analyze_feature_types(df)
        
        # Group features by type and purpose
        self._group_features(df)
        
        # Handle special players
        df = self._handle_special_players(df)
        
        # Apply basic cleanups
        df = self._cleanup_features(df)
        
        # Transform features by group
        df = self._transform_player_features(df)
        df = self._transform_tournament_features(df)
        df = self._transform_performance_features(df)
        df = self._transform_calculated_features(df)
        
        # Add engineered features
        df = self._add_engineered_features(df)
        
        # Final cleanup
        df = self._final_cleanup(df)
        
        return df
    
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:

        df = features_df.copy()
        
        # Apply basic cleanups
        df = self._cleanup_features(df)
        
        # Transform features by group
        df = self._transform_player_features(df)
        df = self._transform_tournament_features(df)
        df = self._transform_performance_features(df)
        df = self._transform_calculated_features(df)
        
        # Add engineered features
        df = self._add_engineered_features(df)
        
        # Final cleanup
        df = self._final_cleanup(df)
        
        return df
    
    def _analyze_feature_types(self, df: pd.DataFrame) -> None:

        for col in df.columns:
            if col in ['player_id', 'tournament_id', 'tournament_id_standard']:
                self.feature_types[col] = 'id'
            elif 'collected_at' in col or '_date' in col:
                self.feature_types[col] = 'datetime'
            elif df[col].dtype == 'object' or df[col].dtype == 'string':
                # Check if it's parseable as numeric
                try:
                    pd.to_numeric(df[col])
                    self.feature_types[col] = 'numeric_string'
                except:
                    # Check if it's a categorial with limited values
                    if df[col].nunique() < 20:
                        self.feature_types[col] = 'categorical'
                    else:
                        self.feature_types[col] = 'text'
            elif np.issubdtype(df[col].dtype, np.number):
                self.feature_types[col] = 'numeric'
            else:
                self.feature_types[col] = 'other'
    
    def _group_features(self, df: pd.DataFrame) -> None:

        self.feature_groups = {
            'ids': [],
            'timestamps': [],
            'player_info': [],
            'tournament_info': [],
            'course_info': [],
            'performance_metrics': [],
            'calculated_metrics': [],
            'position_history': [],
            'strokes_gained': []
        }
        
        # Categorize features into groups
        for col in df.columns:
            # IDs
            if col in ['player_id', 'tournament_id', 'tournament_id_standard']:
                self.feature_groups['ids'].append(col)
            
            # Timestamps
            elif 'collected_at' in col or '_date' in col:
                self.feature_groups['timestamps'].append(col)
            
            # Player info
            elif any(x in col for x in ['name', 'country', 'first_', 'last_', 'full_']):
                self.feature_groups['player_info'].append(col)
            
            # Tournament info
            elif any(x in col for x in ['tournament_', 'tour_code']):
                self.feature_groups['tournament_info'].append(col)
            
            # Course info
            elif any(x in col for x in ['course_', 'overview_', 'par', 'yardage']):
                self.feature_groups['course_info'].append(col)
            
            # Position history
            elif any(x in col for x in ['last1_position', 'last2_position', 'last3_position', 'last4_position', 'last5_position']):
                self.feature_groups['position_history'].append(col)
            
            # Strokes gained
            elif 'sg_' in col or 'strokes_gained' in col:
                self.feature_groups['strokes_gained'].append(col)
            
            # Calculated metrics
            elif any(x in col for x in ['likelihood', 'potential', 'score', 'rating', 'percentile', 'component']):
                self.feature_groups['calculated_metrics'].append(col)
            
            # Performance metrics
            elif self.feature_types[col] == 'numeric':
                self.feature_groups['performance_metrics'].append(col)
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        # Drop timestamp columns if specified
        if self.drop_timestamps:
            timestamp_cols = [col for col in result.columns if 'collected_at' in col]
            result = result.drop(columns=timestamp_cols, errors='ignore')
        
        # Handle special formats and parsing
        result = self._parse_special_format_columns(result)
        
        return result
    
    def _parse_special_format_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        # Parse achievement_cuts_made format (e.g., "339/378")
        if 'achievement_cuts_made' in result.columns:
            try:
                # Extract numerator and denominator
                result['cuts_made_count'] = result['achievement_cuts_made'].str.extract(r'(\d+)/\d+').astype(float)
                result['cuts_made_total'] = result['achievement_cuts_made'].str.extract(r'\d+/(\d+)').astype(float)
                
                # Calculate ratio
                result['cuts_made_ratio'] = result['cuts_made_count'] / result['cuts_made_total'].replace(0, np.nan)
                
                # Drop original column
                result = result.drop(columns=['achievement_cuts_made'])
            except:
                # If parsing fails, keep the original column
                pass
        
        # Parse other complex string columns
        for col in df.columns:
            if self.feature_types.get(col) == 'numeric_string':
                try:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
                except:
                    pass
        
        return result
    
    def _handle_special_players(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self.special_player_ids:
            return df
            
        result = df.copy()
        
        # Add flag for special players
        result['is_special_player'] = result['player_id'].isin(self.special_player_ids).astype(int)
        
        # For future implementations: custom imputation or feature engineering for special players
        
        return result
    
    def _transform_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        # Handle player experience level
        if 'experience_level' in result.columns:
            exp_mapping = {
                'Rookie': 0, 
                'Early Career': 1, 
                'Mid Career': 2, 
                'Veteran': 3
            }
            
            result['experience_level_numeric'] = result['experience_level'].map(exp_mapping)
        
        # Handle career success rating
        if 'career_success_rating' in result.columns:
            rating_mapping = {
                'Below Average': 0,
                'Average': 1,
                'Good': 2,
                'Very Good': 3,
                'Elite': 4
            }
            
            result['career_success_rating_numeric'] = result['career_success_rating'].map(rating_mapping)
        
        # Handle recent form rating
        if 'recent_form_rating' in result.columns:
            form_mapping = {
                'Very Poor': 0,
                'Poor': 1,
                'Average': 2,
                'Good': 3,
                'Excellent': 4
            }
            
            result['recent_form_rating_numeric'] = result['recent_form_rating'].map(form_mapping)
        
        # Handle course history rating
        if 'course_history_rating' in result.columns:
            history_mapping = {
                'Very Poor': 0,
                'Poor': 1,
                'Average': 2,
                'Good': 3,
                'Excellent': 4
            }
            
            result['course_history_rating_numeric'] = result['course_history_rating'].map(history_mapping)
        
        # Handle tournament experience
        if 'tournament_experience' in result.columns:
            tournament_exp_mapping = {
                'First Time': 0,
                'Limited': 1,
                'Moderate': 2,
                'Extensive': 3
            }
            
            result['tournament_experience_numeric'] = result['tournament_experience'].map(tournament_exp_mapping)
        
        # Handle OWGR tier
        if 'owgr_tier' in result.columns:
            # Create encoder for OWGR tier
            if 'owgr_tier' not in self.categorical_encoders:
                self.categorical_encoders['owgr_tier'] = {}
                tiers = sorted(result['owgr_tier'].dropna().unique())
                for i, tier in enumerate(tiers):
                    self.categorical_encoders['owgr_tier'][tier] = i
            
            # Apply encoding
            result['owgr_tier_numeric'] = result['owgr_tier'].map(self.categorical_encoders['owgr_tier'])
        
        return result
    
    def _transform_tournament_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        # Handle position history (last1_position, last2_position, etc.)
        for col in self.feature_groups.get('position_history', []):
            # Convert position strings (e.g., "T5") to numeric
            if col in result.columns:
                result[f'{col}_numeric'] = result[col].apply(
                    lambda x: int(x.replace('T', '')) if isinstance(x, str) and x.startswith('T') 
                    else int(x) if pd.notna(x) and str(x).isdigit() 
                    else np.nan
                )
        
        # Extract year from tournament ID if it matches pattern RYYYY
        if 'tournament_id' in result.columns:
            try:
                result['tournament_year'] = result['tournament_id'].str.extract(r'R(\d{4})').astype(float)
            except:
                pass
        
        return result
    
    def _transform_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:

        result = df.copy()
        sg_category_mapping = {
            'sg_ott': 'Off the Tee',
            'sg_app': 'Approach',
            'sg_atg': 'Around the Green',
            'sg_p': 'Putting',
            'sg_tot': 'Total'
        }
        
        # Convert category string to enum value
        reverse_sg_mapping = {v: i for i, v in enumerate(sg_category_mapping.values())}
        
        if 'history_best_sg_category' in result.columns:
            result['history_best_sg_category_numeric'] = result['history_best_sg_category'].map(reverse_sg_mapping)
        
        if 'history_worst_sg_category' in result.columns:
            result['history_worst_sg_category_numeric'] = result['history_worst_sg_category'].map(reverse_sg_mapping)
        
        # Scale numeric performance features
        performance_cols = [col for col in self.feature_groups.get('performance_metrics', [])
                          if col in result.columns and is_numeric_column(result[col])]
        
        if performance_cols:
            if 'performance' not in self.numerical_scalers:
                self.numerical_scalers['performance'] = StandardScaler()
                # Fit the scaler but don't transform yet
                self.numerical_scalers['performance'].fit(result[performance_cols].fillna(0))
            
            # Create scaled versions of performance metrics
            scaled_data = self.numerical_scalers['performance'].transform(result[performance_cols].fillna(0))
            for i, col in enumerate(performance_cols):
                result[f'{col}_scaled'] = scaled_data[:, i]
        
        # Scale strokes gained features
        sg_cols = [col for col in self.feature_groups.get('strokes_gained', [])
                  if col in result.columns and is_numeric_column(result[col])]
        
        if sg_cols:
            if 'strokes_gained' not in self.numerical_scalers:
                self.numerical_scalers['strokes_gained'] = StandardScaler()
                # Fit the scaler but don't transform yet
                self.numerical_scalers['strokes_gained'].fit(result[sg_cols].fillna(0))
            
            # Create scaled versions of strokes gained metrics
            scaled_data = self.numerical_scalers['strokes_gained'].transform(result[sg_cols].fillna(0))
            for i, col in enumerate(sg_cols):
                result[f'{col}_scaled'] = scaled_data[:, i]
        
        return result
    
    def _transform_calculated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        # Scale calculated metrics
        calc_cols = [col for col in self.feature_groups.get('calculated_metrics', [])
                    if col in result.columns and is_numeric_column(result[col])]
        
        if calc_cols:
            if 'calculated' not in self.numerical_scalers:
                self.numerical_scalers['calculated'] = MinMaxScaler(feature_range=(0, 1))
                # Fit the scaler but don't transform yet
                self.numerical_scalers['calculated'].fit(result[calc_cols].fillna(0))
            
            # Create scaled versions of calculated metrics
            scaled_data = self.numerical_scalers['calculated'].transform(result[calc_cols].fillna(0))
            for i, col in enumerate(calc_cols):
                result[f'{col}_scaled'] = scaled_data[:, i]
        
        return result
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:

        result = df.copy()
        
        # Create recent position momentum feature
        position_numeric_cols = [col for col in result.columns 
                                if col.endswith('_position_numeric') and 'last' in col]
        
        if len(position_numeric_cols) >= 2:
            # Sort columns by recency (last1, last2, etc.)
            position_numeric_cols = sorted(position_numeric_cols, 
                                          key=lambda x: int(re.search(r'last(\d+)', x).group(1)))
            
            # Calculate weighted average of positions
            weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(position_numeric_cols)]
            weights = [w / sum(weights) for w in weights]
            
            result['weighted_recent_position'] = 0
            for col, weight in zip(position_numeric_cols, weights):
                result['weighted_recent_position'] += result[col].fillna(result[col].mean()) * weight
            
            # Calculate momentum (trend between most recent and average of previous)
            if len(position_numeric_cols) >= 2:
                most_recent = result[position_numeric_cols[0]]
                previous = result[position_numeric_cols[1:]]
                avg_previous = previous.mean(axis=1)
                result['position_momentum_engineered'] = avg_previous - most_recent
        
        # Create course fit score if we have both course and player info
        sg_cols = [col for col in result.columns if col.startswith('strokes_gained_') and is_numeric_column(result[col])]
        course_cols = [col for col in result.columns if col.startswith('overview_') and is_numeric_column(result[col])]
        
        if sg_cols and course_cols:
            # Advanced feature engineering would go here
            # For now we'll add a simple placeholder
            result['course_fit_engineered'] = 0.5
        
        # Add historical vs current form comparison
        if 'recent_form_rating_numeric' in result.columns and 'course_history_rating_numeric' in result.columns:
            result['form_vs_history'] = (
                result['recent_form_rating_numeric'] - result['course_history_rating_numeric']
            )
        
        return result
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        # Remove original string columns that have been converted to numeric
        cols_to_drop = []
        for col in result.columns:
            if f'{col}_numeric' in result.columns:
                cols_to_drop.append(col)
        
        # Drop duplicate columns
        for col in result.columns:
            if col.endswith('_1') or col.endswith('_2') or col.endswith('_3'):
                cols_to_drop.append(col)
        
        # Drop timestamp columns if specified
        if self.drop_timestamps:
            timestamp_cols = [col for col in result.columns 
                             if 'collected_at' in col or self.feature_types.get(col) == 'datetime']
            cols_to_drop.extend(timestamp_cols)
        
        # Drop specified columns
        result = result.drop(columns=cols_to_drop, errors='ignore')
        
        return result


def is_numeric_column(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    
    # Check if string column can be converted to numeric
    if series.dtype == 'object' or series.dtype == 'string':
        try:
            pd.to_numeric(series)
            return True
        except:
            return False
    
    return False