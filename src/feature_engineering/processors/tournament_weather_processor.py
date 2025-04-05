# feature_engineering/processors/tournament_weather_processor.py
import pandas as pd
import numpy as np
from ..base import BaseProcessor

class TournamentWeatherProcessor(BaseProcessor):
    """Process tournament weather data to create meaningful features."""
    
    def extract_features(self, tournament_ids=None, player_ids=None, season=None, tournament_id=None):
        """
        Extract and process tournament weather features.
        
        Args:
            tournament_ids: Tournament ID(s) to extract
            player_ids: List of player IDs (not directly used for weather)
            season: Alternative to years parameter
            tournament_id: Alternative to tournament_ids parameter
            
        Returns:
            DataFrame with processed weather features
        """
        # For compatibility with other processors, accept tournament_id parameter
        if tournament_id is not None and tournament_ids is None:
            tournament_ids = tournament_id
            
        # Use season parameter if years is not provided
        years = season
        
        # Extract weather data
        weather_df = self.data_extractor.extract_tournament_weather(
            tournament_ids=tournament_ids,
            years=years
        )
        
        # Extract round-by-round weather data
        weather_by_round_df = self.data_extractor.extract_tournament_weather_by_round(
            tournament_ids=tournament_ids,
            years=years
        )
        
        if weather_df.empty and weather_by_round_df.empty:
            # Create placeholder data when no weather information is found
            placeholder_df = pd.DataFrame({
                'tournament_id': [tournament_ids],
                'has_weather_data': [0],  # Flag indicating no weather data was found
                'avg_temp': [None],
                'avg_humidity': [None],
                'avg_windspeed': [None],
                'total_precip': [None],
                'weather_difficulty': [None]
            })
            return placeholder_df
        
        # Process the data
        features = self._process_weather_features(weather_df, weather_by_round_df)
        
        # Add flag indicating data was found
        if not features.empty:
            features['has_weather_data'] = 1
        
        return features
    
    def _process_weather_features(self, weather_df, weather_by_round_df):
        """
        Process weather data to create features.
        
        Args:
            weather_df: DataFrame with tournament-level weather data
            weather_by_round_df: DataFrame with round-level weather data
            
        Returns:
            DataFrame with processed weather features
        """
        # Start with tournament-level data
        features = pd.DataFrame()
        
        if not weather_df.empty:
            # Select key columns for features
            key_columns = ['tournament_id', 'tournament_name', 'year', 'location']
            avg_columns = [col for col in weather_df.columns if col.startswith('avg_')]
            total_columns = [col for col in weather_df.columns if col.startswith('total_')]
            
            # Create base features
            features = weather_df[key_columns + avg_columns + total_columns].copy()
            
            # Calculate additional metrics
            if 'avg_temp' in features.columns and 'avg_humidity' in features.columns:
                features = self._add_comfort_index(features)
            
            if 'avg_windspeed' in features.columns and 'avg_windgust' in features.columns:
                features = self._add_wind_metrics(features)
            
            if 'total_precip' in features.columns:
                features = self._add_precipitation_metrics(features)
            
            # Calculate weather variability
            if not weather_by_round_df.empty:
                variability = self._calculate_weather_variability(weather_by_round_df)
                if not variability.empty:
                    features = pd.merge(features, variability, on='tournament_id', how='left')
            
            # Add weather difficulty score
            features = self._create_weather_difficulty_score(features)
        
        # If we only have round-level data but no tournament-level data
        elif not weather_by_round_df.empty:
            # Aggregate round-level data to create tournament features
            agg_funcs = {
                'temp': ['mean', 'min', 'max', 'std'],
                'humidity': ['mean', 'max'],
                'windspeed': ['mean', 'max'],
                'windgust': ['mean', 'max'],
                'precip': ['sum', 'max'],
                'cloudcover': ['mean', 'max']
            }
            
            # Only include columns that exist
            valid_agg_funcs = {}
            for col, funcs in agg_funcs.items():
                if col in weather_by_round_df.columns:
                    valid_agg_funcs[col] = funcs
            
            # Group by tournament and aggregate
            if valid_agg_funcs:
                grouped = weather_by_round_df.groupby(['tournament_id', 'tournament_name', 'year', 'location'])
                agg_features = grouped.agg(valid_agg_funcs).reset_index()
                
                # Flatten multi-index columns
                agg_features.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                      for col in agg_features.columns]
                
                # Rename for consistency
                rename_map = {
                    'temp_mean': 'avg_temp',
                    'humidity_mean': 'avg_humidity',
                    'windspeed_mean': 'avg_windspeed',
                    'windgust_mean': 'avg_windgust',
                    'precip_sum': 'total_precip',
                    'cloudcover_mean': 'avg_cloudcover'
                }
                
                agg_features = agg_features.rename(columns=rename_map)
                
                # Build feature set
                features = agg_features.copy()
                
                # Calculate additional metrics
                if 'avg_temp' in features.columns and 'avg_humidity' in features.columns:
                    features = self._add_comfort_index(features)
                
                if 'avg_windspeed' in features.columns and 'avg_windgust' in features.columns:
                    features = self._add_wind_metrics(features)
                
                if 'total_precip' in features.columns:
                    features = self._add_precipitation_metrics(features)
                
                # Add weather difficulty score
                features = self._create_weather_difficulty_score(features)
        
        return features
    
    def _add_comfort_index(self, df):
        """
        Add temperature-humidity comfort indices.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with added comfort metrics
        """
        df_processed = df.copy()
        
        # Calculate Heat Index (for warm temperatures)
        if 'avg_temp' in df_processed.columns and 'avg_humidity' in df_processed.columns:
            # Convert to numeric to ensure calculations work
            temp = pd.to_numeric(df_processed['avg_temp'], errors='coerce')
            humidity = pd.to_numeric(df_processed['avg_humidity'], errors='coerce')
            
            # Calculate heat index for temperatures above 80°F
            mask = temp > 80
            if mask.any():
                # Heat Index formula
                hi = np.where(
                    mask,
                    -42.379 + 2.04901523*temp + 10.14333127*humidity -
                    0.22475541*temp*humidity - 6.83783e-3*temp**2 -
                    5.481717e-2*humidity**2 + 1.22874e-3*temp**2*humidity +
                    8.5282e-4*temp*humidity**2 - 1.99e-6*temp**2*humidity**2,
                    temp
                )
                df_processed['heat_index'] = hi
            else:
                df_processed['heat_index'] = temp
            
            # Calculate Wind Chill for cooler temperatures
            if 'avg_windspeed' in df_processed.columns:
                windspeed = pd.to_numeric(df_processed['avg_windspeed'], errors='coerce')
                
                # Wind chill formula for temperatures below 50°F and wind speeds above 3 mph
                mask = (temp < 50) & (windspeed > 3)
                if mask.any():
                    wc = np.where(
                        mask,
                        35.74 + 0.6215*temp - 35.75*windspeed**0.16 + 0.4275*temp*windspeed**0.16,
                        temp
                    )
                    df_processed['wind_chill'] = wc
                else:
                    df_processed['wind_chill'] = temp
            
            # Calculate "feels like" temperature
            if 'heat_index' in df_processed.columns and 'wind_chill' in df_processed.columns:
                # Use heat index when hot, wind chill when cold, actual temp otherwise
                df_processed['feels_like'] = np.where(
                    temp > 80, df_processed['heat_index'],
                    np.where(temp < 50, df_processed['wind_chill'], temp)
                )
            elif 'heat_index' in df_processed.columns:
                df_processed['feels_like'] = np.where(temp > 80, df_processed['heat_index'], temp)
            elif 'wind_chill' in df_processed.columns:
                df_processed['feels_like'] = np.where(temp < 50, df_processed['wind_chill'], temp)
        
        return df_processed
    
    def _add_wind_metrics(self, df):
        """
        Add derived wind metrics.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with added wind metrics
        """
        df_processed = df.copy()
        
        # Calculate wind metrics
        if 'avg_windspeed' in df_processed.columns:
            windspeed = pd.to_numeric(df_processed['avg_windspeed'], errors='coerce')
            
            # Classify wind conditions
            conditions = [
                (windspeed < 5),
                (windspeed >= 5) & (windspeed < 10),
                (windspeed >= 10) & (windspeed < 15),
                (windspeed >= 15) & (windspeed < 20),
                (windspeed >= 20)
            ]
            
            labels = ['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong']
            
            # Create wind category column - use object type explicitly to avoid category dtype
            df_processed['wind_category'] = np.select(conditions, labels, default='Unknown').astype('object')
            
            # Wind difficulty factor (higher wind speed makes golf more difficult)
            df_processed['wind_difficulty'] = np.clip(windspeed / 5, 0, 5)
            
            # Wind gust differential (if available)
            if 'avg_windgust' in df_processed.columns:
                windgust = pd.to_numeric(df_processed['avg_windgust'], errors='coerce')
                df_processed['gust_differential'] = windgust - windspeed
        
        return df_processed
    
    def _add_precipitation_metrics(self, df):
        """
        Add derived precipitation metrics.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with added precipitation metrics
        """
        df_processed = df.copy()
        
        # Calculate precipitation metrics
        if 'total_precip' in df_processed.columns:
            precip = pd.to_numeric(df_processed['total_precip'], errors='coerce')
            
            # Classify precipitation intensity
            conditions = [
                (precip == 0),
                (precip > 0) & (precip < 0.1),
                (precip >= 0.1) & (precip < 0.5),
                (precip >= 0.5) & (precip < 1),
                (precip >= 1)
            ]
            
            labels = ['None', 'Trace', 'Light', 'Moderate', 'Heavy']
            
            # Create precipitation category column - use object type to avoid category dtype
            df_processed['precip_intensity'] = np.select(conditions, labels, default='Unknown').astype('object')
            
            # Tournament wetness score (0 to a maximum of 5 for very wet conditions)
            df_processed['wetness_score'] = np.clip(precip * 2, 0, 5)
        
        return df_processed
    
    def _calculate_weather_variability(self, weather_by_round_df):
        """
        Calculate weather variability metrics across rounds.
        
        Args:
            weather_by_round_df: DataFrame with round-level weather data
            
        Returns:
            DataFrame with variability metrics by tournament
        """
        # Ensure key columns exist
        if weather_by_round_df.empty or 'tournament_id' not in weather_by_round_df.columns:
            return pd.DataFrame()
        
        # Group by tournament
        grouped = weather_by_round_df.groupby('tournament_id')
        
        variability_metrics = []
        
        for tournament_id, group in grouped:
            metrics = {'tournament_id': tournament_id}
            
            # Calculate temperature variability
            if 'temp' in group.columns:
                temp_values = pd.to_numeric(group['temp'], errors='coerce')
                metrics['temp_variability'] = temp_values.std()
                metrics['temp_range'] = temp_values.max() - temp_values.min()
            
            # Calculate wind variability
            if 'windspeed' in group.columns:
                wind_values = pd.to_numeric(group['windspeed'], errors='coerce')
                metrics['wind_variability'] = wind_values.std()
                metrics['wind_range'] = wind_values.max() - wind_values.min()
            
            # Calculate conditions variability (number of unique conditions)
            if 'conditions' in group.columns:
                metrics['conditions_variability'] = group['conditions'].nunique()
            
            variability_metrics.append(metrics)
        
        return pd.DataFrame(variability_metrics)
    
    def _create_weather_difficulty_score(self, df):
        """
        Create an overall weather difficulty score for each tournament.
        
        Args:
            df: DataFrame with processed weather metrics
            
        Returns:
            DataFrame with added difficulty score
        """
        df_processed = df.copy()
        
        # Start with a base score
        df_processed['weather_difficulty'] = np.zeros(len(df_processed))
        
        # Add contributions from different factors
        # Wind contribution (0-5 points)
        if 'wind_difficulty' in df_processed.columns:
            df_processed['weather_difficulty'] = df_processed['weather_difficulty'] + df_processed['wind_difficulty']
        elif 'avg_windspeed' in df_processed.columns:
            windspeed = pd.to_numeric(df_processed['avg_windspeed'], errors='coerce')
            df_processed['weather_difficulty'] = df_processed['weather_difficulty'] + np.clip(windspeed / 5, 0, 5)
        
        # Temperature extremity contribution (0-3 points)
        if 'avg_temp' in df_processed.columns:
            temp = pd.to_numeric(df_processed['avg_temp'], errors='coerce')
            # High temperature penalty
            high_temp_score = np.where(temp > 85, np.clip((temp - 85) / 5, 0, 3), 0)
            # Low temperature penalty
            low_temp_score = np.where(temp < 55, np.clip((55 - temp) / 10, 0, 3), 0)
            # Use the higher of the two penalties
            df_processed['weather_difficulty'] = df_processed['weather_difficulty'] + np.maximum(high_temp_score, low_temp_score)
        
        # Precipitation contribution (0-5 points)
        if 'wetness_score' in df_processed.columns:
            df_processed['weather_difficulty'] = df_processed['weather_difficulty'] + df_processed['wetness_score']
        elif 'total_precip' in df_processed.columns:
            precip = pd.to_numeric(df_processed['total_precip'], errors='coerce')
            df_processed['weather_difficulty'] = df_processed['weather_difficulty'] + np.clip(precip * 2, 0, 5)
        
        # Precipitation intensity contribution (0-4 points)
        if 'precip_intensity' in df_processed.columns:
            # Define scores for each category - avoid using map with category dtype
            precip_scores = {
                'None': 0.0,
                'Trace': 1.0,
                'Light': 2.0,
                'Moderate': 3.0,
                'Heavy': 4.0,
                'Unknown': 0.0
            }
            
            # Use a safer approach than direct mapping
            intensity_score = np.zeros(len(df_processed))
            for category, score in precip_scores.items():
                intensity_score = np.where(df_processed['precip_intensity'] == category, score, intensity_score)
            
            df_processed['weather_difficulty'] = df_processed['weather_difficulty'] + intensity_score
        
        # Variability contribution (0-3 points)
        if 'temp_variability' in df_processed.columns:
            temp_var = pd.to_numeric(df_processed['temp_variability'], errors='coerce')
            df_processed['weather_difficulty'] = df_processed['weather_difficulty'] + np.clip(temp_var / 10, 0, 1.5)
        
        if 'wind_variability' in df_processed.columns:
            wind_var = pd.to_numeric(df_processed['wind_variability'], errors='coerce')
            df_processed['weather_difficulty'] = df_processed['weather_difficulty'] + np.clip(wind_var / 5, 0, 1.5)
        
        # Normalize to a 0-10 scale
        max_possible_score = 17  # Sum of all possible contributions
        df_processed['weather_difficulty'] = np.clip(df_processed['weather_difficulty'] * 10 / max_possible_score, 0, 10)
        
        # Add a categorical difficulty rating
        difficulty_conditions = [
            (df_processed['weather_difficulty'] < 2),
            (df_processed['weather_difficulty'] >= 2) & (df_processed['weather_difficulty'] < 4),
            (df_processed['weather_difficulty'] >= 4) & (df_processed['weather_difficulty'] < 6),
            (df_processed['weather_difficulty'] >= 6) & (df_processed['weather_difficulty'] < 8),
            (df_processed['weather_difficulty'] >= 8)
        ]
        
        difficulty_labels = ['Easy', 'Moderate', 'Challenging', 'Difficult', 'Extreme']
        
        # Create difficulty category column - use object type to avoid category dtype
        df_processed['weather_difficulty_category'] = np.select(
            difficulty_conditions, 
            difficulty_labels, 
            default='Unknown'
        ).astype('object')
        
        return df_processed