# feature_engineering/processors/course_fit_processor.py
import pandas as pd
import numpy as np
from ..base import BaseProcessor

class CourseFitProcessor(BaseProcessor):
    """Process player course fit data to create meaningful features."""
    
    def extract_features(self, tournament_id=None, player_ids=None, season=None):
        """
        Extract and process course fit features.
        
        Args:
            tournament_id: Tournament ID to extract
            player_ids: List of player IDs to filter by
            season: Season (not directly used for course fit)
            
        Returns:
            DataFrame with processed course fit features
        """
        # Extract course fit data
        course_fit_df = self.data_extractor.extract_course_fit(
            tournament_id=tournament_id,
            player_ids=player_ids
        )
        
        # Even if no data is found, we can create placeholder features
        if course_fit_df.empty and player_ids:
            # Create a dataframe with just player_id and tournament_id
            placeholder_data = []
            for player_id in player_ids:
                placeholder_data.append({
                    'player_id': player_id,
                    'tournament_id': tournament_id,
                    'has_course_fit_data': 0  # Flag indicating no data was found
                })
            features = pd.DataFrame(placeholder_data)
        else:
            # Process the data into features
            features = self._process_course_fit(course_fit_df, tournament_id)
            
            # Add flag indicating data was found
            if not features.empty and 'player_id' in features.columns:
                features['has_course_fit_data'] = 1
        
        return features
    
    def _process_course_fit(self, course_fit_df, tournament_id):
        """
        Process course fit data into meaningful features.
        
        Args:
            course_fit_df: DataFrame with course fit data
            tournament_id: Tournament ID
            
        Returns:
            DataFrame with processed features
        """
        # Create base features with player ID and tournament ID
        features = pd.DataFrame()
        
        if not course_fit_df.empty:
            # Check if we have any player data
            if len(course_fit_df) == 0:
                # No players found, but we can return an empty DataFrame with the tournament_id
                if tournament_id:
                    return pd.DataFrame({'tournament_id': [tournament_id]})
                return pd.DataFrame()
                
            # Extract core columns
            core_cols = ['player_id', 'tournament_id', 'total_rounds', 'score']
            core_cols = [col for col in core_cols if col in course_fit_df.columns]
            
            # Start with basic info
            features = course_fit_df[core_cols].copy()
            
            # Add tournament_id if not present
            if 'tournament_id' not in features.columns and tournament_id:
                features['tournament_id'] = tournament_id
            
            # Process strokes gained categories
            features = self._add_sg_category_features(features, course_fit_df)
            
            # Process ranking metrics
            features = self._add_ranking_features(features, course_fit_df)
            
            # Calculate overall course fit score
            features = self._add_overall_fit_score(features, course_fit_df)
        elif tournament_id:
            # If no data but we have a tournament ID, return a DataFrame with just the tournament ID
            features = pd.DataFrame({'tournament_id': [tournament_id]})
        
        return features
    
    def _add_sg_category_features(self, features, course_fit_df):
        """
        Add features based on strokes gained categories.
        
        Args:
            features: DataFrame to update
            course_fit_df: Raw course fit data
            
        Returns:
            Updated features DataFrame
        """
        # Find all value columns
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
        
        # Calculate specialized fit metrics based on column patterns
        
        # Putting distance metrics
        putting_distance_cols = {
            'short_putt': [col for col in value_cols if 'putting' in col and any(d in col for d in ['0_8ft', '0-8ft', 'short'])],
            'medium_putt': [col for col in value_cols if 'putting' in col and any(d in col for d in ['8_20ft', '8-20ft', 'medium'])],
            'long_putt': [col for col in value_cols if 'putting' in col and any(d in col for d in ['20ft', 'long'])]
        }
        
        for distance, cols in putting_distance_cols.items():
            if cols:
                features[f'{distance}_avg'] = course_fit_df[cols].mean(axis=1)
        
        # Approach shot conditions
        approach_condition_cols = {
            'fairway_approach': [col for col in value_cols if 'approach' in col and 'fairway' in col],
            'rough_approach': [col for col in value_cols if 'approach' in col and 'rough' in col]
        }
        
        for condition, cols in approach_condition_cols.items():
            if cols:
                features[f'{condition}_avg'] = course_fit_df[cols].mean(axis=1)
        
        # Around the green conditions
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
        """
        Add features based on player rankings.
        
        Args:
            features: DataFrame to update
            course_fit_df: Raw course fit data
            
        Returns:
            Updated features DataFrame
        """
        # Find all rank columns
        rank_cols = [col for col in course_fit_df.columns if col.endswith('_rank')]
        
        # Add individual rank values
        for col in rank_cols:
            if col in course_fit_df.columns:
                features[col] = course_fit_df[col]
        
        # Group columns by skill area (similar to values)
        skill_areas = {
            'putting': [col for col in rank_cols if 'putting' in col],
            'approach': [col for col in rank_cols if 'approach' in col],
            'around_green': [col for col in rank_cols if 'around_the_green' in col],
            'off_tee': [col for col in rank_cols if 'off_the_tee' in col]
        }
        
        # Calculate aggregate metrics for each skill area
        for area, cols in skill_areas.items():
            if cols:
                # Average rank across all metrics in this skill area
                features[f'{area}_rank_avg'] = course_fit_df[cols].mean(axis=1)
                
                # Best rank in this skill area
                features[f'{area}_rank_best'] = course_fit_df[cols].min(axis=1)  # Lower rank is better
                
                # Worst rank in this skill area
                features[f'{area}_rank_worst'] = course_fit_df[cols].max(axis=1)  # Higher rank is worse
                
                # Count of good ranks (top 50)
                features[f'{area}_top50_count'] = course_fit_df[cols].le(50).sum(axis=1)
                
                # Count of poor ranks (bottom 50)
                total_players = course_fit_df[cols].max().max() if not course_fit_df[cols].empty else 0
                if total_players > 100:  # Only if we have enough players
                    threshold = total_players - 50
                    features[f'{area}_bottom50_count'] = course_fit_df[cols].ge(threshold).sum(axis=1)
        
        # Calculate overall rank metrics
        if rank_cols:
            # Average rank across all metrics
            features['overall_rank_avg'] = course_fit_df[rank_cols].mean(axis=1)
            
            # Best rank across all metrics
            features['overall_rank_best'] = course_fit_df[rank_cols].min(axis=1)
            
            # Worst rank across all metrics
            features['overall_rank_worst'] = course_fit_df[rank_cols].max(axis=1)
            
            # Rank variability (std dev of ranks)
            features['rank_variability'] = course_fit_df[rank_cols].std(axis=1)
            
            # Count of top 25% ranks
            # This requires estimating the field size
            field_size = course_fit_df[rank_cols].max().max() if not course_fit_df[rank_cols].empty else 0
            if field_size > 0:
                q1_threshold = field_size // 4
                features['top_quartile_count'] = course_fit_df[rank_cols].le(q1_threshold).sum(axis=1)
        
        return features
    
    def _add_overall_fit_score(self, features, course_fit_df):
        """
        Calculate overall course fit score.
        
        Args:
            features: DataFrame to update
            course_fit_df: Raw course fit data
            
        Returns:
            Updated features DataFrame
        """
        # Find all value columns
        value_cols = [col for col in course_fit_df.columns if col.endswith('_value')]
        
        # Calculate weighted course fit score
        if value_cols:
            # Basic average of all SG values
            features['avg_sg_value'] = course_fit_df[value_cols].mean(axis=1)
            
            # Sum of all positive SG values (strengths for this course)
            positive_values = course_fit_df[value_cols].copy()
            positive_values[positive_values < 0] = 0
            features['total_positive_sg'] = positive_values.sum(axis=1)
            
            # Count of positive SG values
            features['positive_sg_count'] = course_fit_df[value_cols].gt(0).sum(axis=1)
            
            # Sum of all negative SG values (weaknesses for this course)
            negative_values = course_fit_df[value_cols].copy()
            negative_values[negative_values > 0] = 0
            features['total_negative_sg'] = negative_values.sum(axis=1)
            
            # Count of negative SG values
            features['negative_sg_count'] = course_fit_df[value_cols].lt(0).sum(axis=1)
            
            # Calculate fit ratio (strengths vs weaknesses)
            # Higher is better - more strengths than weaknesses
            features['strength_weakness_ratio'] = features['positive_sg_count'] / (features['negative_sg_count'] + 0.001)
            
            # Calculate course fit score (0-100 scale)
            # This is a simple weighted formula that places more importance on strengths
            # and less negative impact from weaknesses
            
            # Normalize the positive and negative SG totals
            max_pos = features['total_positive_sg'].max()
            min_neg = features['total_negative_sg'].min()
            
            # Calculate normalized values (0-1 scale)
            norm_pos = features['total_positive_sg'] / max_pos if max_pos > 0 else 0
            norm_neg = (features['total_negative_sg'] - min_neg) / (-min_neg) if min_neg < 0 else 0
            
            # Calculate fit score (70% weight on strengths, 30% on lack of weaknesses)
            features['course_fit_score'] = (70 * norm_pos + 30 * (1 - norm_neg)) 
            
            # Scale to 0-100
            features['course_fit_score'] = features['course_fit_score'].clip(0, 100)
            
            # Add categorical fit rating
            conditions = [
                (features['course_fit_score'] >= 80),
                (features['course_fit_score'] >= 60) & (features['course_fit_score'] < 80),
                (features['course_fit_score'] >= 40) & (features['course_fit_score'] < 60),
                (features['course_fit_score'] >= 20) & (features['course_fit_score'] < 40),
                (features['course_fit_score'] < 20)
            ]
            
            choices = ['Excellent', 'Good', 'Average', 'Poor', 'Very Poor']
            
            features['fit_rating'] = np.select(conditions, choices, default='Unknown')
            features['fit_rating'] = features['fit_rating'].astype(str)  # Ensure string type
        
        # Add score-based fit metric if available
        if 'score' in course_fit_df.columns:
            # Higher score means better fit in this dataset
            features['raw_fit_score'] = course_fit_df['score']
            
            # Normalize to 0-100 scale
            min_score = course_fit_df['score'].min()
            max_score = course_fit_df['score'].max()
            
            if max_score > min_score:
                features['normalized_fit_score'] = ((course_fit_df['score'] - min_score) / 
                                                  (max_score - min_score) * 100)
        
        return features