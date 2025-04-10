import pandas as pd
import numpy as np
from ..base import BaseProcessor

class TournamentWeatherProcessor(BaseProcessor):
    def extract_features(self, tournament_ids=None, player_ids=None, season=None, tournament_id=None):
        if tournament_id is not None and tournament_ids is None:
            tournament_ids = tournament_id

        years = season

        weather_df = self.data_extractor.extract_tournament_weather(
            tournament_ids=tournament_ids,
            years=years
        )

        weather_by_round_df = self.data_extractor.extract_tournament_weather_by_round(
            tournament_ids=tournament_ids,
            years=years
        )
        
        if weather_df.empty and weather_by_round_df.empty:
            placeholder_df = pd.DataFrame({
                'tournament_id': [tournament_ids],
                'has_weather_data': [0], 
                'avg_temp': [None],
                'avg_humidity': [None],
                'avg_windspeed': [None],
                'total_precip': [None],
                'weather_difficulty': [None]
            })
            return placeholder_df

        features = self._process_weather_features(weather_df, weather_by_round_df)
        if not features.empty:
            features['has_weather_data'] = 1
        return features
    
def _process_weather_features(self, weather_df, weather_by_round_df):
    features = pd.DataFrame()
    
    if not weather_df.empty:
        keep_columns = ['tournament_id', 'year', 'avg_temp', 'avg_windspeed', 'total_precip']
        features = weather_df[keep_columns].copy()

        features = self._create_weather_difficulty_score(features)

    elif not weather_by_round_df.empty:
        agg_funcs = {
            'temp': 'mean',
            'windspeed': 'mean',
            'precip': 'sum'
        }

        valid_cols = [col for col in agg_funcs if col in weather_by_round_df.columns]
        if valid_cols:
            grouped = weather_by_round_df.groupby(['tournament_id', 'year'])
            agg_features = grouped.agg({col: agg_funcs[col] for col in valid_cols}).reset_index()
            rename_map = {
                'temp': 'avg_temp',
                'windspeed': 'avg_windspeed',
                'precip': 'total_precip'
            }
            agg_features = agg_features.rename(columns=rename_map)
            features = agg_features.copy()

            features = self._create_weather_difficulty_score(features)

    return features

    
    
    def _add_wind_metrics(self, df):
        df_processed = df.copy()
    
        if 'avg_windspeed' in df_processed.columns:
            windspeed = pd.to_numeric(df_processed['avg_windspeed'], errors='coerce')
            conditions = [
                (windspeed < 5),
                (windspeed >= 5) & (windspeed < 10),
                (windspeed >= 10) & (windspeed < 15),
                (windspeed >= 15) & (windspeed < 20),
                (windspeed >= 20)
            ]
            
            labels = ['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong']
     
            df_processed['wind_category'] = np.select(conditions, labels, default='Unknown').astype('object')

            df_processed['wind_difficulty'] = np.clip(windspeed / 5, 0, 5)

            if 'avg_windgust' in df_processed.columns:
                windgust = pd.to_numeric(df_processed['avg_windgust'], errors='coerce')
                df_processed['gust_differential'] = windgust - windspeed
        
        return df_processed
    
    def _add_precipitation_metrics(self, df):
        df_processed = df.copy()
        
        if 'total_precip' in df_processed.columns:
            precip = pd.to_numeric(df_processed['total_precip'], errors='coerce')

            conditions = [
                (precip == 0),
                (precip > 0) & (precip < 0.1),
                (precip >= 0.1) & (precip < 0.5),
                (precip >= 0.5) & (precip < 1),
                (precip >= 1)
            ]
            
            labels = ['None', 'Trace', 'Light', 'Moderate', 'Heavy']

            df_processed['precip_intensity'] = np.select(conditions, labels, default='Unknown').astype('object')
            
            # Tournament wetness score (0 to a maximum of 5 for very wet conditions)
            df_processed['wetness_score'] = np.clip(precip * 2, 0, 5)
        
        return df_processed
    
    
    def _create_weather_difficulty_score(self, df):
        df_processed = df.copy()
        df_processed['weather_difficulty'] = np.zeros(len(df_processed))
        if 'wind_difficulty' in df_processed.columns:
            df_processed['weather_difficulty'] = df_processed['weather_difficulty'] + df_processed['wind_difficulty']
        elif 'avg_windspeed' in df_processed.columns:
            windspeed = pd.to_numeric(df_processed['avg_windspeed'], errors='coerce')
            df_processed['weather_difficulty'] = df_processed['weather_difficulty'] + np.clip(windspeed / 5, 0, 5)

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
    
        max_possible_score = 17  # Sum of all possible contributions
        df_processed['weather_difficulty'] = np.clip(df_processed['weather_difficulty'] * 10 / max_possible_score, 0, 10)
        
        difficulty_conditions = [
            (df_processed['weather_difficulty'] < 2),
            (df_processed['weather_difficulty'] >= 2) & (df_processed['weather_difficulty'] < 4),
            (df_processed['weather_difficulty'] >= 4) & (df_processed['weather_difficulty'] < 6),
            (df_processed['weather_difficulty'] >= 6) & (df_processed['weather_difficulty'] < 8),
            (df_processed['weather_difficulty'] >= 8)
        ]
        
        difficulty_labels = ['Easy', 'Moderate', 'Challenging', 'Difficult', 'Extreme']
        
        df_processed['weather_difficulty_category'] = np.select(
            difficulty_conditions, 
            difficulty_labels, 
            default='Unknown'
        ).astype('object')
        
        return df_processed