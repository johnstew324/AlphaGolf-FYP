# feature_engineering/feature_selection/enhanced_feature_refiner.py

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import GradientBoostingRegressor
from collections import defaultdict

class EnhancedFeatureRefiner:
    """
    Advanced feature refinement with improved handling of missing data,
    multicollinearity, and target-specific differentiation.
    """
    
    def __init__(self, analyzer, selector):
        """
        Initialize the enhanced feature refiner.
        
        Args:
            analyzer: FeatureAnalyzer instance with pre-computed analysis
            selector: FeatureSelector instance for feature selection
        """
        self.analyzer = analyzer
        self.selector = selector
        self.feature_sets = {}
        self.interaction_features = {}
        self.target_specific_features = {
            'win': set(),
            'cut': set(),
            'top3': set(),
            'top10': set()
        }
        self.core_features = set()
        self.group_representatives = {}
        self.features_to_exclude = set()
        
    def identify_features_to_exclude(self, missing_threshold=50, low_variance_threshold=0.01):
        """
        Identify features that should be excluded from all feature sets.
        
        Args:
            missing_threshold: Missing percentage threshold to exclude features
            low_variance_threshold: Variance threshold to exclude features
            
        Returns:
            Set of features to exclude
        """
        # Get features with high missing values
        high_missing = set(self.analyzer.analysis_results.get('high_missing_features', []))
        
        # Get features with low variance
        low_variance = set(self.analyzer.analysis_results.get('low_variance_features', []))
        
        # Combine sets
        to_exclude = high_missing.union(low_variance)
        
        # Manually exclude non-predictive features
        manual_exclude = {
            'scoring_scoring_average_adjusted_total_strokes',
            'par5_to_par_x', 'par4_to_par_x', 'par3_to_par_x',
            'course_yards_per_par', 'course_overview_yardage',
            'course_yardage', 'total_score_std_form',
            'score_std_form', 'yards_per_par'
        }
        
        to_exclude.update(manual_exclude)
        
        # Always keep certain features regardless of statistical properties
        always_keep = {
            'player_id', 'tournament_id', 'owgr', 'win_rate', 
            'history_sg_app', 'history_sg_tot', 'fit_component'
        }
        
        # Remove always-keep features from exclusion list
        self.features_to_exclude = {f for f in to_exclude if f not in always_keep}
        
        return self.features_to_exclude
        
    def select_target_specific_core_features(self):
        """
        Select core features for each prediction target based on 
        statistical properties and domain knowledge.
        
        Returns:
            Dict of target-specific core feature sets
        """
        # Define domain-specific feature groups important for each target
        domain_features = {
            'win': [
                # Key features for win prediction
                'history_sg_app', 'history_sg_short_game', 'recent_top10',
                'history_best_sg_value', 'win_rate', 'top_10_pct',
                'recent_avg_score', 'scoring_variability', 'history_sg_atg',
                'fit_component', 'strokes_gained_approach_the_green_other_value_scaled'
            ],
            'cut': [
                # Key features for cut prediction (focus on consistency)
                'cuts_made', 'consistency_ratio', 'par4_bogey_pct',
                'pars_pct', 'scoring_variability', 'worst_total_score_history',
                'par4_avg', 'par3_avg', 'history_sg_p', 'back_nine_par_pct'
            ],
            'top3': [
                # Key features for top3 prediction (blend of win and top10)
                'history_sg_app', 'top_25_pct', 'win_rate',
                'front_nine_birdie_pct', 'recent_finish_std', 'strokes_gained_approach_the_green_other_value_scaled',
                'birdies', 'avg_finish_history', 'approach_rank_best'
            ],
            'top10': [
                # Key features for top10 prediction
                'top_10_pct', 'avg_finish_history', 'recent_top10',
                'history_sg_tot', 'par5_birdie_pct', 'top_25_pct',
                'scoring_variability', 'cuts_made_pct', 'history_sg_app'
            ]
        }
        
        # Get importance analysis for feature selection
        importance_data = {}
        for target in ['win', 'cut', 'top3', 'top10']:
            target_col = self._map_target_name_to_column(target)
            if f'importance_{target_col}' in self.analyzer.analysis_results:
                importance_df = self.analyzer.analysis_results[f'importance_{target_col}']
                importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
                importance_data[target] = importance_dict
        
        target_core_features = {}
        for target, core_list in domain_features.items():
            # Add statistical top features if available
            statistical_features = set()
            if target in importance_data:
                # Get top 30 features by importance
                sorted_features = sorted(importance_data[target].items(), 
                                        key=lambda x: x[1], reverse=True)
                top_features = [f for f, _ in sorted_features[:30] 
                               if f not in self.features_to_exclude]
                statistical_features.update(top_features)
            
            # Combine domain knowledge features with statistical features
            combined = set(core_list).union(statistical_features)
            
            # Remove excluded features
            filtered = {f for f in combined if f not in self.features_to_exclude}
            
            target_core_features[target] = list(filtered)
        
        return target_core_features
    
    def handle_multicollinearity(self, correlation_threshold=0.8):
        """
        Identify highly correlated feature groups and select representatives.
        Uses both correlation analysis and the multicollinearity report.
        
        Args:
            correlation_threshold: Threshold to identify high correlations
            
        Returns:
            Dict of group representative features
        """
        # Get correlated groups from analyzer
        correlated_groups = self.analyzer.analysis_results.get('correlated_feature_groups', [])
        
        # Add manually identified correlated groups from multicollinearity report
        manual_groups = [
            # Group 1: Win metrics
            ['wins', 'win_rate', 'career_win_rate', 'victory_potential'],
            
            # Group 2: Strokes gained putting and short game
            ['history_sg_p', 'history_sg_short_game'],
            
            # Group 3: Strokes gained percentile groups
            ['history_sg_app_pct', 'history_sg_atg_pct', 'history_sg_ott_pct', 'history_sg_p_pct'],
            
            # Group 4: Scoring trends
            ['first_to_last_diff', 'score_trend'],
            
            # Group 5: Win rate variations
            ['win_rate', 'wins_1_scaled', 'career_wins', 'international_wins']
        ]
        
        for group in manual_groups:
            # Only add group if it's not already covered
            if not any(set(group).issubset(set(existing)) for existing in correlated_groups):
                correlated_groups.append(group)
        
        # Representatives from each group
        representatives = {}
        
        # Get missing values information
        missing_dict = {}
        if 'missing_values' in self.analyzer.analysis_results:
            missing_df = self.analyzer.analysis_results['missing_values']
            missing_dict = dict(zip(missing_df['feature'], missing_df['missing_pct']))
        
        # Process each group to select representatives
        for i, group in enumerate(correlated_groups):
            group_name = f"group_{i+1}"
            
            # Skip empty groups or those with only one feature
            if not group or len(group) < 2:
                continue
            
            # Calculate quality scores for each feature
            feature_scores = []
            for feature in group:
                # Skip features that should be excluded
                if feature in self.features_to_exclude:
                    continue
                
                # Calculate score components
                missing_pct = missing_dict.get(feature, 0)
                name_length = len(feature)
                
                # Prefer simpler named features with lower missing values
                # Higher score is better
                score = (100 - missing_pct) / (name_length + 1)
                
                # Prefer recent/current features over historical ones
                if any(p in feature for p in ['recent_', 'current_']):
                    score *= 1.2
                
                # Prefer core metrics over derived ones
                if any(p in feature for p in ['_std', '_scaled', '_pct']):
                    score *= 0.8
                
                feature_scores.append((feature, score))
            
            # Sort by score and select best feature(s)
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top 1-2 features depending on group size
            num_to_select = min(2, max(1, len(group) // 10))
            selected = [item[0] for item in feature_scores[:num_to_select]]
            
            if selected:  # Only add non-empty groups
                representatives[group_name] = selected
        
        self.group_representatives = representatives
        return representatives
    
    def create_optimized_target_feature_sets(self):
        """
        Create optimized feature sets for each prediction target,
        accounting for multicollinearity and missing data.
        
        Returns:
            Dict of optimized feature sets by target
        """
        # Get target-specific core features
        target_core_features = self.select_target_specific_core_features()
        
        # Get group representatives for multicollinearity
        self.handle_multicollinearity()
        
        # Create optimized feature sets
        optimized_sets = {}
        
        for target, core_features in target_core_features.items():
            # Start with core features for this target
            selected = set(core_features)
            
            # Add representative features from correlated groups
            for group_features in self.group_representatives.values():
                # Add all representatives (typically 1-2 per group)
                selected.update(group_features)
            
            # Add ID columns
            selected.add('player_id')
            selected.add('tournament_id')
            
            # Add target-specific interaction features
            target_interactions = self.create_target_specific_interactions(target, list(selected))
            selected.update(target_interactions.keys())
            
            # Store the optimized set
            optimized_sets[target] = list(selected)
            
            # Store interactions for this target
            self.target_specific_features[target] = selected
            self.interaction_features.update(target_interactions)
        
        return optimized_sets
    
    def create_target_specific_interactions(self, target, base_features):
        """
        Create target-specific interaction features.
        
        Args:
            target: Target type ('win', 'cut', 'top3', 'top10')
            base_features: Base features to consider for interactions
            
        Returns:
            Dict of interaction features
        """
        interactions = {}
        
        # Define target-specific feature patterns for interactions
        target_patterns = {
            'win': [
                # Pairs that matter for win prediction
                ('history_sg_app', ['approach_rank_best', 'strokes_gained_approach_the_green_other_value_scaled', 'fit_component']),
                ('history_sg_short_game', ['approach_rank_best', 'strokes_gained_approach_the_green_other_value_scaled', 'fit_component']),
                ('history_sg_atg', ['approach_rank_best', 'strokes_gained_approach_the_green_other_value_scaled', 'fit_component'])
            ],
            'cut': [
                # Pairs that matter for cut prediction
                ('par4_avg', ['par4_bogey_pct', 'consistency_ratio']),
                ('history_sg_p', ['putting_rank_best', 'putting_rank_worst']),
                ('worst_total_score_history', ['scoring_variability', 'pars_pct'])
            ],
            'top3': [
                # Pairs that matter for top3 prediction
                ('history_sg_app', ['approach_rank_best', 'fit_component']),
                ('top_25_pct', ['recent_top10', 'recent_finish_std']),
                ('front_nine_birdie_pct', ['back_nine_par_pct', 'par5_birdie_pct'])
            ],
            'top10': [
                # Pairs that matter for top10 prediction
                ('top_10_pct', ['recent_top10', 'avg_finish_history']),
                ('history_sg_tot', ['scoring_variability', 'top_25_pct']),
                ('par5_birdie_pct', ['par4_avg', 'par3_avg'])
            ]
        }
        
        # Select patterns for the current target
        current_patterns = target_patterns.get(target, [])
        
        # Create interactions based on patterns
        for feat1, feat2_list in current_patterns:
            if feat1 not in base_features:
                continue
                
            for feat2 in feat2_list:
                if feat2 not in base_features:
                    continue
                
                # Create interaction feature name
                interaction_name = f"interaction_{feat1}_{feat2}"
                
                # Register the interaction
                interactions[interaction_name] = {
                    'feature1': feat1,
                    'feature2': feat2,
                    'type': 'product'
                }
        
        return interactions
    
    def save_optimized_feature_sets(self, output_path):
        """
        Save the optimized feature sets to a JSON file.
        
        Args:
            output_path: Path to save the feature sets
            
        Returns:
            None
        """
        # Create optimized feature sets if not already done
        if not hasattr(self, 'optimized_sets'):
            self.optimized_sets = self.create_optimized_target_feature_sets()
        
        # Create dictionary of feature sets
        feature_sets = {
            'core_features': list(self.core_features),
            'group_representatives': self.group_representatives,
            'target_specific_features': {k: list(v) for k, v in self.target_specific_features.items()},
            'interaction_features': {k: v for k, v in self.interaction_features.items()},
            'optimized_sets': self.optimized_sets
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(feature_sets, f, indent=2)
        
        return feature_sets
    
    def _map_target_name_to_column(self, target_name):
        """Map target names to actual column names."""
        mapping = {
            'win': 'winner',
            'cut': 'made_cut',
            'top3': 'top3',
            'top10': 'top10',
            'position': 'position'
        }
        return mapping.get(target_name, target_name)

# Usage example:
"""
# Initialize analyzer and selector
analyzer = FeatureAnalyzer(features_df, target_df)
analyzer.analyze_features()

# Initialize selector
selector = FeatureSelector(features_df, analyzer, target_df)

# Initialize enhanced refiner
refiner = EnhancedFeatureRefiner(analyzer, selector)

# Identify features to exclude
refiner.identify_features_to_exclude()

# Create optimized feature sets
optimized_sets = refiner.create_optimized_target_feature_sets()

# Save optimized feature sets
refiner.save_optimized_feature_sets('feature_engineering/feature_refinement/optimized_feature_sets.json')
"""