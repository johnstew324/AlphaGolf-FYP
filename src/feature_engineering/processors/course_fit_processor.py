# feature_engineering/processors/course_fit_processor.py
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
            placeholder_data = []
            for player_id in player_ids:
                placeholder_data.append({
                    'player_id': player_id,
                    'tournament_id': tournament_id,
                    'has_course_fit_data': 0  
                })
            features = pd.DataFrame(placeholder_data)
        else:
            features = self._process_course_fit(course_fit_df, tournament_id)
            
            if not features.empty and 'player_id' in features.columns:
                features['has_course_fit_data'] = 1
        
        return features
    
    def _process_course_fit(self, course_fit_df, tournament_id):
        features = pd.DataFrame()
        if not course_fit_df.empty:
            if len(course_fit_df) == 0:
            
                if tournament_id:
                    return pd.DataFrame({'tournament_id': [tournament_id]})
                return pd.DataFrame()
                
            core_cols = ['player_id', 'tournament_id', 'total_rounds', 'score']
            core_cols = [col for col in core_cols if col in course_fit_df.columns]
            
            features = course_fit_df[core_cols].copy()
            
            if 'tournament_id' not in features.columns and tournament_id:
                features['tournament_id'] = tournament_id
            features = self._add_sg_category_features(features, course_fit_df)
            
            features = self._add_ranking_features(features, course_fit_df)
            
            features = self._add_overall_fit_score(features, course_fit_df)
        elif tournament_id:
            features = pd.DataFrame({'tournament_id': [tournament_id]})
        
        return features
    
    def _add_sg_category_features(self, features, course_fit_df):
        value_cols = [col for col in course_fit_df.columns if col.endswith('_value')]
        
        # Group columns by skill area
        skill_areas = {
            'putting': [col for col in value_cols if 'putting' in col],
            'approach': [col for col in value_cols if 'approach' in col],
            'around_green': [col for col in value_cols if 'around_the_green' in col],
            'off_tee': [col for col in value_cols if 'off_the_tee' in col]
        }
        
        # Add individual SG values
        for col in value_cols:
            if col in course_fit_df.columns:
                features[col] = course_fit_df[col]
        
        # Calculate aggregate metrics for each skill area
        for area, cols in skill_areas.items():
            if cols:
                # Average value across all metrics in this skill area
                features[f'{area}_avg'] = course_fit_df[cols].mean(axis=1)
                
                # Best value in this skill area
                features[f'{area}_best'] = course_fit_df[cols].max(axis=1)
                
                # Worst value in this skill area
                features[f'{area}_worst'] = course_fit_df[cols].min(axis=1)
                
                # Count of positive values (strengths)
                features[f'{area}_strengths'] = course_fit_df[cols].gt(0).sum(axis=1)
                
                # Count of negative values (weaknesses)
                features[f'{area}_weaknesses'] = course_fit_df[cols].lt(0).sum(axis=1)
        
        putting_distance_cols = {
            'short_putt': [col for col in value_cols if 'putting' in col and any(d in col for d in ['0_8ft', '0-8ft', 'short'])],
            'medium_putt': [col for col in value_cols if 'putting' in col and any(d in col for d in ['8_20ft', '8-20ft', 'medium'])],
            'long_putt': [col for col in value_cols if 'putting' in col and any(d in col for d in ['20ft', 'long'])]
        }
        
        for distance, cols in putting_distance_cols.items():
            if cols:
                features[f'{distance}_avg'] = course_fit_df[cols].mean(axis=1)
        approach_condition_cols = {
            'fairway_approach': [col for col in value_cols if 'approach' in col and 'fairway' in col],
            'rough_approach': [col for col in value_cols if 'approach' in col and 'rough' in col]
        }
        
        for condition, cols in approach_condition_cols.items():
            if cols:
                features[f'{condition}_avg'] = course_fit_df[cols].mean(axis=1)

        around_green_cols = {
            'bunker_play': [col for col in value_cols if 'around_the_green' in col and 'bunker' in col],
            'rough_scrambling': [col for col in value_cols if 'around_the_green' in col and 'rough' in col],
            'fringe_play': [col for col in value_cols if 'around_the_green' in col and ('fringe' in col or 'fairway' in col)]
        }
        
        for condition, cols in around_green_cols.items():
            if cols:
                features[f'{condition}_avg'] = course_fit_df[cols].mean(axis=1)
        
        return features
    
    def _add_ranking_features(self, features, course_fit_df):
        rank_cols = [col for col in course_fit_df.columns if col.endswith('_rank')]
        
        for col in rank_cols:
            if col in course_fit_df.columns:
                features[col] = course_fit_df[col]

        skill_areas = {
            'putting': [col for col in rank_cols if 'putting' in col],
            'approach': [col for col in rank_cols if 'approach' in col],
            'around_green': [col for col in rank_cols if 'around_the_green' in col],
            'off_tee': [col for col in rank_cols if 'off_the_tee' in col]
        }

        for area, cols in skill_areas.items():
            if cols:
                features[f'{area}_rank_avg'] = course_fit_df[cols].mean(axis=1)
                
                features[f'{area}_rank_best'] = course_fit_df[cols].min(axis=1) 
                
                features[f'{area}_rank_worst'] = course_fit_df[cols].max(axis=1)  
                
                features[f'{area}_top50_count'] = course_fit_df[cols].le(50).sum(axis=1)
                
                total_players = course_fit_df[cols].max().max() if not course_fit_df[cols].empty else 0
                if total_players > 100:
                    threshold = total_players - 50
                    features[f'{area}_bottom50_count'] = course_fit_df[cols].ge(threshold).sum(axis=1)
        
        if rank_cols:

            features['overall_rank_avg'] = course_fit_df[rank_cols].mean(axis=1)
            features['overall_rank_best'] = course_fit_df[rank_cols].min(axis=1)
            features['overall_rank_worst'] = course_fit_df[rank_cols].max(axis=1)
            features['rank_variability'] = course_fit_df[rank_cols].std(axis=1)
            field_size = course_fit_df[rank_cols].max().max() if not course_fit_df[rank_cols].empty else 0
            if field_size > 0:
                q1_threshold = field_size // 4
                features['top_quartile_count'] = course_fit_df[rank_cols].le(q1_threshold).sum(axis=1)
        
        return features
    
    def _add_overall_fit_score(self, features, course_fit_df):
        value_cols = [col for col in course_fit_df.columns if col.endswith('_value')]
        
        if value_cols:
            features['avg_sg_value'] = course_fit_df[value_cols].mean(axis=1)
            
            positive_values = course_fit_df[value_cols].copy()
            positive_values[positive_values < 0] = 0
            features['total_positive_sg'] = positive_values.sum(axis=1)
            
            features['positive_sg_count'] = course_fit_df[value_cols].gt(0).sum(axis=1)

            negative_values = course_fit_df[value_cols].copy()
            negative_values[negative_values > 0] = 0
            features['total_negative_sg'] = negative_values.sum(axis=1)
            features['negative_sg_count'] = course_fit_df[value_cols].lt(0).sum(axis=1)

            features['strength_weakness_ratio'] = features['positive_sg_count'] / (features['negative_sg_count'] + 0.001)
            
            max_pos = features['total_positive_sg'].max()
            min_neg = features['total_negative_sg'].min()
            
            norm_pos = features['total_positive_sg'] / max_pos if max_pos > 0 else 0
            norm_neg = (features['total_negative_sg'] - min_neg) / (-min_neg) if min_neg < 0 else 0
            
            features['course_fit_score'] = (70 * norm_pos + 30 * (1 - norm_neg)) 
            
            features['course_fit_score'] = features['course_fit_score'].clip(0, 100)
            
            conditions = [
                (features['course_fit_score'] >= 80),
                (features['course_fit_score'] >= 60) & (features['course_fit_score'] < 80),
                (features['course_fit_score'] >= 40) & (features['course_fit_score'] < 60),
                (features['course_fit_score'] >= 20) & (features['course_fit_score'] < 40),
                (features['course_fit_score'] < 20)
            ]
            
            choices = ['Excellent', 'Good', 'Average', 'Poor', 'Very Poor']
            
            features['fit_rating'] = np.select(conditions, choices, default='Unknown')
            features['fit_rating'] = features['fit_rating'].astype(str) 
        
        if 'score' in course_fit_df.columns:
            features['raw_fit_score'] = course_fit_df['score']

            min_score = course_fit_df['score'].min()
            max_score = course_fit_df['score'].max()
            
            if max_score > min_score:
                features['normalized_fit_score'] = ((course_fit_df['score'] - min_score) / 
                                                  (max_score - min_score) * 100)
        
        return features