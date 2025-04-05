# feature_engineering/processors/course_stats_processor.py
import pandas as pd
import numpy as np
from ..base import BaseProcessor

class CourseStatsProcessor(BaseProcessor):
    """Process course statistics data to create meaningful features."""
    
    def extract_features(self, tournament_id=None, player_ids=None, season=None):
        """
        Extract and process course statistics features.
        
        Args:
            tournament_id: Tournament ID to extract
            player_ids: List of player IDs (not directly used for course stats)
            season: Season to extract (derived from tournament_id)
            
        Returns:
            DataFrame with processed course statistics features
        """
        # Extract year from tournament_id if possible
        year = None
        if tournament_id and len(tournament_id) >= 5 and tournament_id[0] == 'R':
            try:
                year = int(tournament_id[1:5])
            except ValueError:
                # If conversion fails, use season parameter
                year = season if season else None
        elif season:
            year = season
        
        # Extract course-level statistics
        course_stats_df = self.data_extractor.extract_course_stats(
            tournament_ids=tournament_id,
            #years=year
        )
        
        # Extract hole-level statistics
        hole_stats_df = self.data_extractor.extract_hole_stats(
            tournament_ids=tournament_id,
            #years=year
        )
        
        if course_stats_df.empty and hole_stats_df.empty:
            return pd.DataFrame()
        
        # Process the data into features
        features = self._process_course_stats(course_stats_df, hole_stats_df, tournament_id)
        
        return features
    
    def _process_course_stats(self, course_stats_df, hole_stats_df, tournament_id):
        """
        Process course statistics into meaningful features.
        
        Args:
            course_stats_df: DataFrame with course-level statistics
            hole_stats_df: DataFrame with hole-level statistics
            tournament_id: Tournament ID
            
        Returns:
            DataFrame with processed course features
        """
        # Initialize features DataFrame
        features = pd.DataFrame()
        
        # Process course-level statistics if available
        if not course_stats_df.empty:
            course_features = self._extract_course_features(course_stats_df)
            features = course_features.copy()
        
        # Process hole-level statistics if available
        if not hole_stats_df.empty:
            hole_features = self._extract_hole_features(hole_stats_df)
            
            # If we already have course features, merge them
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
        
        # If we have data, add tournament_id as a feature
        if not features.empty and 'tournament_id' not in features.columns:
            features['tournament_id'] = tournament_id
        
        return features
    
    def _extract_course_features(self, course_stats_df):
        """
        Extract course-level features from course statistics.
        
        Args:
            course_stats_df: DataFrame with course-level statistics
            
        Returns:
            DataFrame with course-level features
        """
        # Select one row per tournament (in case multiple courses)
        if 'host_course' in course_stats_df.columns:
            # Prefer host course if available
            host_courses = course_stats_df[course_stats_df['host_course'] == True]
            if not host_courses.empty:
                course_df = host_courses.copy()
            else:
                course_df = course_stats_df.copy()
        else:
            course_df = course_stats_df.copy()
        
        # If multiple courses per tournament, take first one
        course_df = course_df.drop_duplicates('tournament_id', keep='first')
        
        # Select relevant columns for features
        core_cols = [
            'tournament_id', 'course_id', 'course_name', 'par', 'yardage', 'year'
        ]
        
        # Add overview columns if available
        overview_cols = [col for col in course_df.columns if col.startswith('overview_')]
        
        # Add summary columns if available
        summary_cols = [col for col in course_df.columns if col.startswith('summary_')]
        
        # Extract selected columns
        selected_cols = core_cols + overview_cols + summary_cols
        selected_cols = [col for col in selected_cols if col in course_df.columns]
        
        course_features = course_df[selected_cols].copy()
        
        # Calculate derived features
        self._add_derived_course_features(course_features)
        
        return course_features
    
    def _add_derived_course_features(self, course_features):
        """
        Add derived course features.
        
        Args:
            course_features: DataFrame to update with derived features
        """
        # Convert numerical overview values
        for col in ['overview_par', 'overview_yardage']:
            if col in course_features.columns:
                try:
                    course_features[col.replace('overview_', '')] = pd.to_numeric(
                        course_features[col], errors='coerce'
                    )
                except:
                    pass
        
        # Calculate par distribution if summary data exists
        if all(col in course_features.columns for col in ['summary_eagles', 'summary_birdies', 'summary_pars', 'summary_bogeys', 'summary_double_bogeys']):
            total_scores = (
                course_features['summary_eagles'] + 
                course_features['summary_birdies'] + 
                course_features['summary_pars'] + 
                course_features['summary_bogeys'] + 
                course_features['summary_double_bogeys']
            )
            
            # Calculate percentages
            course_features['eagles_pct'] = course_features['summary_eagles'] / total_scores * 100
            course_features['birdies_pct'] = course_features['summary_birdies'] / total_scores * 100
            course_features['pars_pct'] = course_features['summary_pars'] / total_scores * 100
            course_features['bogeys_pct'] = course_features['summary_bogeys'] / total_scores * 100
            course_features['doubles_pct'] = course_features['summary_double_bogeys'] / total_scores * 100
            
            # Calculate under/over par ratio
            under_par = course_features['summary_eagles'] + course_features['summary_birdies']
            over_par = course_features['summary_bogeys'] + course_features['summary_double_bogeys']
            
            course_features['under_over_ratio'] = under_par / over_par
            
            # Calculate overall scoring difficulty (higher is more difficult)
            weighted_score = (
                course_features['eagles_pct'] * -2 +
                course_features['birdies_pct'] * -1 +
                course_features['pars_pct'] * 0 +
                course_features['bogeys_pct'] * 1 +
                course_features['doubles_pct'] * 2
            )
            
            course_features['scoring_difficulty'] = weighted_score
        
        # Add established year feature
        if 'overview_established' in course_features.columns:
            try:
                established = pd.to_numeric(course_features['overview_established'], errors='coerce')
                if 'year' in course_features.columns:
                    course_features['course_age'] = course_features['year'] - established
            except:
                pass
        
        # Calculate yardage per par
        if 'yardage' in course_features.columns and 'par' in course_features.columns:
            course_features['yards_per_par'] = course_features['yardage'] / course_features['par']
        
        # Extract course record if available
        if 'overview_record' in course_features.columns:
            try:
                course_features['course_record'] = pd.to_numeric(
                    course_features['overview_record'], errors='coerce'
                )
            except:
                pass
    
    def _extract_hole_features(self, hole_stats_df):
        """
        Extract hole-level features from hole statistics.
        
        Args:
            hole_stats_df: DataFrame with hole-level statistics
            
        Returns:
            DataFrame with aggregated hole features
        """
        # Group by tournament to create aggregated features
        hole_features = []
        
        for tournament_id, tournament_holes in hole_stats_df.groupby('tournament_id'):
            # Base feature record
            feature_record = {'tournament_id': tournament_id}
            
            # Calculate hole type statistics
            feature_record.update(self._calculate_hole_type_stats(tournament_holes))
            
            # Calculate difficulty statistics
            feature_record.update(self._calculate_difficulty_stats(tournament_holes))
            
            # Calculate pin positions statistics
            feature_record.update(self._calculate_pin_position_stats(tournament_holes))
            
            # Calculate round statistics if multiple rounds exist
            rounds = tournament_holes['round_number'].unique()
            if len(rounds) > 1:
                feature_record.update(self._calculate_round_progression(tournament_holes))
            
            hole_features.append(feature_record)
        
        return pd.DataFrame(hole_features)
    
    def _calculate_hole_type_stats(self, holes_df):
        """
        Calculate statistics by hole type (par 3, 4, 5).
        
        Args:
            holes_df: DataFrame with hole data
            
        Returns:
            Dictionary with hole type statistics
        """
        stats = {}
        
        # Group holes by par
        par_groups = holes_df.groupby('hole_par')
        
        # Calculate statistics for each par type
        for par, par_holes in par_groups:
            par_key = f'par{par}'
            
            # Count holes of this par
            stats[f'{par_key}_count'] = len(par_holes)
            
            # Average length
            if 'hole_yards' in par_holes.columns:
                stats[f'{par_key}_avg_length'] = par_holes['hole_yards'].mean()
            
            # Average scoring
            if 'hole_scoring_average' in par_holes.columns:
                stats[f'{par_key}_scoring_avg'] = par_holes['hole_scoring_average'].mean()
                stats[f'{par_key}_to_par'] = par_holes['hole_scoring_average'].mean() - par
            
            # Scoring distribution
            for score_type in ['eagles', 'birdies', 'pars', 'bogeys', 'double_bogeys']:
                col = f'hole_{score_type}'
                if col in par_holes.columns:
                    total = par_holes[col].sum()
                    stats[f'{par_key}_{score_type}'] = total
                    # Calculate percentage if we have the hole count
                    total_opportunities = par_holes[col].sum() * par_holes['hole_par'].count()
                    if total_opportunities > 0:
                        stats[f'{par_key}_{score_type}_pct'] = (total / total_opportunities) * 100
        
        # Overall par 3-4-5 performance differential
        if all(f'par{p}_to_par' in stats for p in [3, 4, 5]):
            # Compare par 3 performance to par 4-5 average
            par45_avg = (stats['par4_to_par'] + stats['par5_to_par']) / 2
            stats['par3_differential'] = stats['par3_to_par'] - par45_avg
            
            # Compare par 5 scoring to expected
            stats['par5_scoring_advantage'] = -1 * stats['par5_to_par']  # Convert to strokes gained
        
        return stats
    
    def _calculate_difficulty_stats(self, holes_df):
        """
        Calculate hole difficulty statistics.
        
        Args:
            holes_df: DataFrame with hole data
            
        Returns:
            Dictionary with difficulty statistics
        """
        stats = {}
        
        # Skip if missing key columns
        if 'hole_scoring_average' not in holes_df.columns or 'hole_par' not in holes_df.columns:
            return stats
        
        # Calculate relative difficulty (scoring average relative to par)
        holes_df['relative_difficulty'] = holes_df['hole_scoring_average'] - holes_df['hole_par']
        
        # Calculate overall course difficulty
        stats['avg_difficulty'] = holes_df['relative_difficulty'].mean()
        
        # Calculate difficulty variability
        stats['difficulty_std'] = holes_df['relative_difficulty'].std()
        
        # Identify most difficult and easiest holes
        if 'hole_rank' in holes_df.columns:
            # Lower rank = more difficult
            easiest = holes_df.nsmallest(3, 'hole_rank')
            hardest = holes_df.nlargest(3, 'hole_rank')
            
            # Average difficulty of extreme holes
            stats['hardest_holes_avg'] = hardest['relative_difficulty'].mean()
            stats['easiest_holes_avg'] = easiest['relative_difficulty'].mean()
            
            # Difficulty spread
            stats['difficulty_range'] = stats['hardest_holes_avg'] - stats['easiest_holes_avg']
        else:
            # Use relative difficulty if rank not available
            easiest = holes_df.nsmallest(3, 'relative_difficulty')
            hardest = holes_df.nlargest(3, 'relative_difficulty')
            
            stats['hardest_holes_avg'] = hardest['relative_difficulty'].mean()
            stats['easiest_holes_avg'] = easiest['relative_difficulty'].mean()
            stats['difficulty_range'] = stats['hardest_holes_avg'] - stats['easiest_holes_avg']
        
        # Calculate front/back nine differential
        front_nine = holes_df[holes_df['hole_number'] <= 9]
        back_nine = holes_df[holes_df['hole_number'] > 9]
        
        if not front_nine.empty and not back_nine.empty:
            stats['front_nine_difficulty'] = front_nine['relative_difficulty'].mean()
            stats['back_nine_difficulty'] = back_nine['relative_difficulty'].mean()
            stats['nine_difficulty_diff'] = stats['back_nine_difficulty'] - stats['front_nine_difficulty']
        
        return stats
    
    def _calculate_pin_position_stats(self, holes_df):
        """
        Calculate pin position statistics.
        
        Args:
            holes_df: DataFrame with hole data
            
        Returns:
            Dictionary with pin position statistics
        """
        stats = {}
        
        # Check if pin position data exists
        pin_cols = [col for col in holes_df.columns if col.startswith('pin_')]
        if not pin_cols:
            return stats
        
        # Calculate average pin positions
        for axis in ['left_to_right', 'bottom_to_top']:
            for coord in ['x', 'y']:
                col = f'pin_{axis}_{coord}'
                if col in holes_df.columns:
                    stats[f'avg_{col}'] = holes_df[col].mean()
        
        # Calculate pin position clustering
        for axis in ['left_to_right', 'bottom_to_top']:
            for coord in ['x', 'y']:
                col = f'pin_{axis}_{coord}'
                if col in holes_df.columns:
                    stats[f'{col}_std'] = holes_df[col].std()
        
        # Calculate pin position difficulty correlation
        # Higher values mean more difficult pins correlate with more difficult holes
        if 'relative_difficulty' in holes_df.columns:
            for axis in ['left_to_right', 'bottom_to_top']:
                for coord in ['x', 'y']:
                    col = f'pin_{axis}_{coord}'
                    if col in holes_df.columns:
                        try:
                            corr = holes_df[col].corr(holes_df['relative_difficulty'])
                            stats[f'{col}_difficulty_corr'] = corr
                        except:
                            pass
        
        return stats
    
    def _calculate_round_progression(self, holes_df):
        """
        Calculate round-to-round progression statistics.
        
        Args:
            holes_df: DataFrame with hole data
            
        Returns:
            Dictionary with round progression statistics
        """
        stats = {}
        
        # Group by round
        round_groups = holes_df.groupby('round_number')
        
        # Track key metrics by round
        round_metrics = {}
        
        for round_num, round_holes in round_groups:
            # Skip if missing key columns
            if 'hole_scoring_average' not in round_holes.columns or 'hole_par' not in round_holes.columns:
                continue
                
            # Calculate relative difficulty
            rel_diff = round_holes['hole_scoring_average'] - round_holes['hole_par']
            round_metrics[round_num] = {
                'difficulty': rel_diff.mean(),
                'eagles': round_holes['hole_eagles'].sum() if 'hole_eagles' in round_holes.columns else 0,
                'birdies': round_holes['hole_birdies'].sum() if 'hole_birdies' in round_holes.columns else 0,
                'bogeys': round_holes['hole_bogeys'].sum() if 'hole_bogeys' in round_holes.columns else 0,
                'doubles': round_holes['hole_double_bogeys'].sum() if 'hole_double_bogeys' in round_holes.columns else 0
            }
        
        # Calculate round-to-round changes
        round_nums = sorted(round_metrics.keys())
        if len(round_nums) > 1:
            # Calculate difficulty progression
            difficulties = [round_metrics[r]['difficulty'] for r in round_nums]
            
            # Overall trend (negative = getting easier, positive = getting harder)
            if len(difficulties) >= 2:
                try:
                    stats['difficulty_trend'] = np.polyfit(range(len(difficulties)), difficulties, 1)[0]
                except:
                    pass
            
            # First to final round comparison
            if round_nums[0] in round_metrics and round_nums[-1] in round_metrics:
                first = round_metrics[round_nums[0]]
                final = round_metrics[round_nums[-1]]
                
                stats['first_to_final_diff'] = final['difficulty'] - first['difficulty']
                
                # Compare scoring distribution
                if all(key in first and key in final for key in ['eagles', 'birdies', 'bogeys', 'doubles']):
                    first_under = first['eagles'] + first['birdies']
                    final_under = final['eagles'] + final['birdies']
                    first_over = first['bogeys'] + first['doubles']
                    final_over = final['bogeys'] + final['doubles']
                    
                    # Positive means more under-par scores in final round
                    stats['final_rd_under_par_diff'] = final_under - first_under
                    # Positive means more over-par scores in final round
                    stats['final_rd_over_par_diff'] = final_over - first_over
        
        return stats