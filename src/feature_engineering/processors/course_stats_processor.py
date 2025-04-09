# feature_engineering/processors/course_stats_processor.py
import pandas as pd
import numpy as np
from ..base import BaseProcessor

class CourseStatsProcessor(BaseProcessor):
    def extract_features(self, tournament_id=None, player_ids=None, season=None):
        year = None
        if tournament_id and len(tournament_id) >= 5 and tournament_id[0] == 'R':
            try:
                year = int(tournament_id[1:5])
            except ValueError:
                year = season if season else None
        elif season:
            year = season
        course_stats_df = self.data_extractor.extract_course_stats(
            tournament_ids=tournament_id,
        )
        hole_stats_df = self.data_extractor.extract_hole_stats(
            tournament_ids=tournament_id,
        )
        
        if course_stats_df.empty and hole_stats_df.empty:
            return pd.DataFrame()
        features = self._process_course_stats(course_stats_df, hole_stats_df, tournament_id)

        return features
    
    def _process_course_stats(self, course_stats_df, hole_stats_df, tournament_id):
        features = pd.DataFrame()

        if not course_stats_df.empty:
            course_features = self._extract_course_features(course_stats_df)
            features = course_features.copy()

        if not hole_stats_df.empty:
            hole_features = self._extract_hole_features(hole_stats_df)
            if not features.empty:
                features = pd.merge(
                    features, 
                    hole_features,
                    on='tournament_id',
                    how='outer',
                    suffixes=('', '_hole')
                )
            else:
                features = hole_features.copy()

        if not features.empty and 'tournament_id' not in features.columns:
            features['tournament_id'] = tournament_id
        
        return features
    
    def _extract_course_features(self, course_stats_df):
        core_cols = [
            'tournament_id', 'course_id', 'course_name', 'par', 'yardage', 'year'
        ]
        overview_cols = [col for col in course_stats_df.columns if col.startswith('overview_')]
        summary_cols = [col for col in course_stats_df.columns if col.startswith('summary_')]

        selected_cols = core_cols + overview_cols + summary_cols
        selected_cols = [col for col in selected_cols if col in course_stats_df.columns]

        course_features = course_stats_df[selected_cols].copy()
        
        # Only keep core scoring metrics like eagles, birdies, pars, bogeys, and double bogeys
        if all(col in course_features.columns for col in ['summary_eagles', 'summary_birdies', 'summary_pars', 'summary_bogeys', 'summary_double_bogeys']):
            total_scores = (
                course_features['summary_eagles'] + 
                course_features['summary_birdies'] + 
                course_features['summary_pars'] + 
                course_features['summary_bogeys'] + 
                course_features['summary_double_bogeys']
            )
            
            course_features['eagles_pct'] = course_features['summary_eagles'] / total_scores * 100
            course_features['birdies_pct'] = course_features['summary_birdies'] / total_scores * 100
            course_features['pars_pct'] = course_features['summary_pars'] / total_scores * 100
            course_features['bogeys_pct'] = course_features['summary_bogeys'] / total_scores * 100
            course_features['doubles_pct'] = course_features['summary_double_bogeys'] / total_scores * 100

            under_par = course_features['summary_eagles'] + course_features['summary_birdies']
            over_par = course_features['summary_bogeys'] + course_features['summary_double_bogeys']
            
            course_features['under_over_ratio'] = under_par / over_par

        return course_features
    
    def _extract_hole_features(self, hole_stats_df):
        hole_features = []
        
        for tournament_id, tournament_holes in hole_stats_df.groupby('tournament_id'):
            feature_record = {'tournament_id': tournament_id}
        
            feature_record.update(self._calculate_difficulty_stats(tournament_holes))
            hole_features.append(feature_record)
        
        return pd.DataFrame(hole_features)
    
    def _calculate_difficulty_stats(self, holes_df):
        stats = {}
        
        if 'hole_scoring_average' not in holes_df.columns or 'hole_par' not in holes_df.columns:
            return stats
        holes_df['relative_difficulty'] = holes_df['hole_scoring_average'] - holes_df['hole_par']

        stats['avg_difficulty'] = holes_df['relative_difficulty'].mean()
        stats['difficulty_range'] = holes_df['relative_difficulty'].max() - holes_df['relative_difficulty'].min()

        return stats
