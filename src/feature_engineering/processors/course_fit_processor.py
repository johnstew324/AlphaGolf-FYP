import pandas as pd
import numpy as np
from ..base import BaseProcessor

class CourseFitProcessor(BaseProcessor):
    def extract_features(self, tournament_id=None, player_ids=None, season=None):
        course_fit_df = self.data_extractor.extract_course_fit(
            tournament_id=tournament_id,
            player_ids=player_ids
        )

        if course_fit_df.empty and player_ids:
            placeholder_data = [
                {
                    'player_id': player_id,
                    'tournament_id': tournament_id,
                    'has_course_fit_data': 0
                }
                for player_id in player_ids
            ]
            return pd.DataFrame(placeholder_data)

        features = self._process_course_fit(course_fit_df, tournament_id)
        if not features.empty and 'player_id' in features.columns:
            features['has_course_fit_data'] = 1
        return features

    def _process_course_fit(self, course_fit_df, tournament_id):
        if course_fit_df.empty:
            return pd.DataFrame({'tournament_id': [tournament_id]}) if tournament_id else pd.DataFrame()

        features = pd.DataFrame()
        core_cols = ['player_id', 'tournament_id', 'total_rounds']
        features = course_fit_df[core_cols].copy()

        if 'tournament_id' not in features.columns and tournament_id:
            features['tournament_id'] = tournament_id

        features = self._add_sg_category_averages(features, course_fit_df)
        features = self._add_overall_rank_features(features, course_fit_df)

        return features

    def _add_sg_category_averages(self, features, course_fit_df):
        value_cols = [col for col in course_fit_df.columns if col.endswith('_value')]
        sg_areas = {
            'putting': [col for col in value_cols if 'putting' in col],
            'approach': [col for col in value_cols if 'approach' in col],
            'around_green': [col for col in value_cols if 'around_the_green' in col],
            'off_tee': [col for col in value_cols if 'off_the_tee' in col]
        }

        for area, cols in sg_areas.items():
            if cols:
                features[f'{area}_avg'] = course_fit_df[cols].mean(axis=1)

        return features

    def _add_overall_rank_features(self, features, course_fit_df):
        rank_cols = [col for col in course_fit_df.columns if col.endswith('_rank')]
        if not rank_cols:
            return features

        features['overall_rank_avg'] = course_fit_df[rank_cols].mean(axis=1)
        features['rank_variability'] = course_fit_df[rank_cols].std(axis=1)

        field_size = course_fit_df[rank_cols].max().max() if not course_fit_df[rank_cols].empty else 0
        if field_size > 0:
            top_quartile = field_size // 4
            features['top_quartile_count'] = course_fit_df[rank_cols].le(top_quartile).sum(axis=1)

        return features
