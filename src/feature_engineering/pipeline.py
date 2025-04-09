# feature_engineering/pipeline.py
import pandas as pd
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from .processors.player_stats_processor import PlayerFormProcessor
from .processors.course_fit_processor import CourseFitProcessor
from .processors.tournament_history_processor import TournamentHistoryProcessor
from .processors.player_profile_overview_processor import PlayerProfileProcessor
from .processors.player_career_processor import PlayerCareerProcessor
from .processors.scorecard_processor import ScorecardProcessor
from .processors.tournament_weather_processor import TournamentWeatherProcessor
from .processors.course_stats_processor import CourseStatsProcessor
from .processors.current_form_processor import CurrentFormProcessor
from .processors.tournament_history_stats_processor import TournamentHistoryStatsProcessor

class FeaturePipeline:
    
    def __init__(self, data_extractor):
        self.data_extractor = data_extractor
        self.player_form = PlayerFormProcessor(data_extractor)
        self.course_fit = CourseFitProcessor(data_extractor)
        self.tournament_history = TournamentHistoryProcessor(data_extractor)
        self.player_profile = PlayerProfileProcessor(data_extractor)
        self.player_career = PlayerCareerProcessor(data_extractor)
        self.scorecard = ScorecardProcessor(data_extractor)
        self.tournament_weather = TournamentWeatherProcessor(data_extractor)
        self.course_stats = CourseStatsProcessor(data_extractor)
        self.current_form = CurrentFormProcessor(data_extractor)
        self.tournament_history_stats = TournamentHistoryStatsProcessor(data_extractor)

    def generate_features(self, tournament_id, season, player_ids=None):
        if tournament_id.startswith("R") and len(tournament_id) >= 8:
            special_tournament_id = "R2025" + tournament_id[5:]
        else:
            special_tournament_id = tournament_id

        player_form = self.player_form.extract_features(player_ids, season, tournament_id)

        course_fit = self.course_fit.extract_features(tournament_id, player_ids)
        
        tournament_history = self.tournament_history.extract_features(special_tournament_id, player_ids)
        
        # Get player profile features
        player_profile = self.player_profile.extract_features(player_ids, season, tournament_id)
        
        # Get player career features
        player_career = self.player_career.extract_features(player_ids, season, tournament_id)
        
        # Get scorecard features (using standard format)
        scorecard = self.scorecard.extract_features(tournament_id, player_ids, season)
        
        course_stats = self.course_stats.extract_features(tournament_id, player_ids, season)

        weather_features = self.tournament_weather.extract_features(tournament_ids=tournament_id, season=season)
        
        current_form_features = self.current_form.extract_features(player_ids, season, tournament_id)
        
        tournament_history_stats = self.tournament_history_stats.extract_features(tournament_id, player_ids, season)

        features = self._combine_features(player_form, course_fit, tournament_history, player_profile, player_career, scorecard, weather_features, course_stats, current_form_features, tournament_history_stats)
        
        return features
    
    def _combine_features(self, *feature_sets):
        result = None
        
        for i, features in enumerate(feature_sets):
            if features is None or features.empty:
                print(f"Feature set {i} is empty or None")
                continue

            print(f"Feature set {i}: {features.shape} columns, example columns: {list(features.columns)[:5]}")
            
            if result is not None:
                print(f"After merge {i}: {result.shape} columns")
    
    def generate_target_variables(self, tournament_id, player_ids=None):
        if tournament_id.startswith("R") and len(tournament_id) >= 8:
            special_id = "R2025" + tournament_id[5:]
        else:
            special_id = tournament_id
        history = self.data_extractor.extract_tournament_history(
            tournament_ids=special_id,
            player_ids=player_ids
        )
        
        if history.empty:
            return pd.DataFrame()
        
        print(f"Tournament history columns: {history.columns.tolist()}")

        targets = []
        
        player_id_col = None
        for possible_col in ['player_id', 'pid', 'player', 'id']:
            if possible_col in history.columns:
                player_id_col = possible_col
                break
                
        if player_id_col is None:
            print(f"Error: No player ID column found in tournament history data")
            return pd.DataFrame()
            
        for _, player_data in history.iterrows():
            player_target = {
                'player_id': player_data[player_id_col],  
                'tournament_id': tournament_id,
                'year': player_data.get('year')
            }
            
            if 'position_numeric' in player_data:
                pos = player_data['position_numeric']
                if pd.notna(pos):
                    player_target['position'] = pos
                    player_target['winner'] = 1 if pos == 1 else 0
                    player_target['top3'] = 1 if pos <= 3 else 0
                    player_target['top10'] = 1 if pos <= 10 else 0
                    player_target['made_cut'] = 1 if pos < 100 else 0
            
            targets.append(player_target)

        targets_df = pd.DataFrame(targets)

        for col in ['winner', 'top3', 'top10', 'made_cut']:
            if col in targets_df.columns:
                targets_df[col] = targets_df[col].astype(int)

        targets_df['history_tournament_id'] = special_id
        
        return targets_df