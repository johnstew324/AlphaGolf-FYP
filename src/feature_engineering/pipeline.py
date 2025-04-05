# feature_engineering/pipeline.py
import pandas as pd
import numpy as np
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
    """Main pipeline for feature engineering."""
    
    def __init__(self, data_extractor):
        """
        Initialize the pipeline with a data extractor.
        
        Args:
            data_extractor: An instance of the DataExtractor class
        """
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
        """
        Generate features for a specific tournament.
        
        Args:
            tournament_id: The tournament ID in standard RYYYY format
            season: The current season
            player_ids: Optional list of player IDs
            
        Returns:
            DataFrame with combined features
        """
        # Create special tournament ID format for tournament_history
        # Convert from standard RYYYY format to special R2025 format
        if tournament_id.startswith("R") and len(tournament_id) >= 8:
            special_tournament_id = "R2025" + tournament_id[5:]
        else:
            special_tournament_id = tournament_id
        
        # Get player form features
        player_form = self.player_form.extract_features(player_ids, season, tournament_id)
        
        # Get course fit features
        course_fit = self.course_fit.extract_features(tournament_id, player_ids)
        
        # Get tournament history features (using special format)
        tournament_history = self.tournament_history.extract_features(special_tournament_id, player_ids)
        
        # Get player profile features
        player_profile = self.player_profile.extract_features(player_ids, season, tournament_id)
        
        # Get player career features
        player_career = self.player_career.extract_features(player_ids, season, tournament_id)
        
        # Get scorecard features (using standard format)
        scorecard = self.scorecard.extract_features(tournament_id, player_ids, season)
        
        course_stats = self.course_stats.extract_features(tournament_id, player_ids, season)
        
        # Get weather features
        weather_features = self.tournament_weather.extract_features(tournament_ids=tournament_id, season=season)
        
        current_form_features = self.current_form.extract_features(player_ids, season, tournament_id)
        
        tournament_history_stats = self.tournament_history_stats.extract_features(tournament_id, player_ids, season)
        
        # Combine all features
        features = self._combine_features(player_form, course_fit, tournament_history, player_profile, player_career, scorecard, weather_features, course_stats, current_form_features, tournament_history_stats)
        
        return features
    
    def _combine_features(self, *feature_sets):
        """
        Combine multiple feature sets into a single DataFrame.
        
        Args:
            *feature_sets: Multiple DataFrames to combine
            
        Returns:
            Combined DataFrame
        """
        # Start with an empty result
        result = None
        
        for features in feature_sets:
            if features is None or features.empty:
                continue
                
            # Ensure the features DataFrame has player_id column
            if 'player_id' not in features.columns:
                print(f"Warning: Feature set missing player_id column, skipping")
                continue
                
            if result is None:
                # First non-empty DataFrame becomes the base
                result = features.copy()
            else:
                # Merge with existing results on player_id
                try:
                    result = pd.merge(result, features, on='player_id', how='outer')
                except Exception as e:
                    print(f"Error merging feature sets: {str(e)}")
                    # Print column information for debugging
                    print(f"Left columns: {result.columns.tolist()}")
                    print(f"Right columns: {features.columns.tolist()}")
        
        return result if result is not None else pd.DataFrame()
    
    def generate_target_variables(self, tournament_id, player_ids=None):
        """
        Generate target variables for model training.
        
        Args:
            tournament_id: The tournament ID in standard RYYYY format
            player_ids: Optional list of player IDs
            
        Returns:
            DataFrame with target variables
        """
        # Convert the standard tournament_id to the special format used in tournament_history
        if tournament_id.startswith("R") and len(tournament_id) >= 8:
            special_id = "R2025" + tournament_id[5:]
        else:
            special_id = tournament_id
        
        # Extract tournament results using the special ID format
        history = self.data_extractor.extract_tournament_history(
            tournament_ids=special_id,
            player_ids=player_ids
        )
        
        if history.empty:
            return pd.DataFrame()
        
        # Debug: Print column names to see what's available
        print(f"Tournament history columns: {history.columns.tolist()}")
        
        # Create target variables
        targets = []
        
        # Determine what column holds player IDs (could be 'player_id', 'player', 'pid', etc.)
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
                'player_id': player_data[player_id_col],  # Use the identified column
                'tournament_id': tournament_id,
                'year': player_data.get('year')
            }
            
            # Position-based targets
            if 'position_numeric' in player_data:
                pos = player_data['position_numeric']
                if pd.notna(pos):
                    player_target['position'] = pos
                    player_target['winner'] = 1 if pos == 1 else 0
                    # ...rest of the function remains unchanged...
                    player_target['top3'] = 1 if pos <= 3 else 0
                    player_target['top10'] = 1 if pos <= 10 else 0
                    player_target['made_cut'] = 1 if pos < 100 else 0
            
            targets.append(player_target)
        
        # Create DataFrame and ensure proper types
        targets_df = pd.DataFrame(targets)
        
        # Convert target columns to integers where appropriate
        for col in ['winner', 'top3', 'top10', 'made_cut']:
            if col in targets_df.columns:
                targets_df[col] = targets_df[col].astype(int)
        
        # Add a column indicating the special tournament_id used for lookup
        targets_df['history_tournament_id'] = special_id
        
        return targets_df