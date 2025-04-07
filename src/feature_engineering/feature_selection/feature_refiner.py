# feature_engineering/feature_selection/feature_refiner.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FeatureRefiner:
    """
    Advanced feature refinement for golf tournament prediction.
    
    This class implements strategies to reduce dimensionality, handle multicollinearity,
    and create optimized feature sets for different prediction targets.
    """
    
    def __init__(self, analyzer, selector):
        """
        Initialize the feature refiner.
        
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
        
        # Core features that should be considered for all targets
        self.core_features = set()
        
        # Feature group representatives to avoid multicollinearity
        self.group_representatives = {}
        
    def identify_core_features(self, importance_threshold=0.5, correlation_threshold=0.8, max_features=50):
        """
        Identify core features that should be considered for all prediction targets.
        
        Args:
            importance_threshold: Minimum importance score to consider
            correlation_threshold: Maximum correlation allowed between features
            max_features: Maximum number of core features to select
            
        Returns:
            Set of core feature names
        """
        # Get feature importance results if available
        if 'importance_victory_potential' in self.analyzer.analysis_results:
            importance_df = self.analyzer.analysis_results['importance_victory_potential']
            
            # Get top features by importance
            top_features = importance_df[importance_df['importance'] >= importance_threshold]['feature'].tolist()
            
            # Filter for correlation
            selected_features = []
            correlation_matrix = self.analyzer.analysis_results.get('correlation_matrix', 
                                                                   pd.DataFrame())
            
            # Add features one by one, checking correlation with already selected
            for feature in top_features:
                # Skip if not in correlation matrix
                if feature not in correlation_matrix.columns:
                    continue
                    
                # Check correlation with already selected features
                should_add = True
                for selected in selected_features:
                    if selected in correlation_matrix.columns and feature in correlation_matrix.index:
                        corr = abs(correlation_matrix.loc[feature, selected])
                        if corr > correlation_threshold:
                            should_add = False
                            break
                
                if should_add:
                    selected_features.append(feature)
                    
                # Stop if we reached max features
                if len(selected_features) >= max_features:
                    break
                    
            self.core_features = set(selected_features)
            return self.core_features
        else:
            # If no importance analysis is available, use selector to pick features
            selected_df = self.selector.select_features(method='combined')
            self.core_features = set(selected_df.columns[:max_features])
            return self.core_features
    
    def select_group_representatives(self, correlated_groups, target_col=None):
        """
        Select representative features from correlated groups.
        
        Args:
            correlated_groups: List of lists containing correlated feature groups
            target_col: Optional target column to select features by importance
            
        Returns:
            Dict mapping group names to selected representative features
        """
        representatives = {}
        
        # If we have target information and importance analysis, use it
        importance_dict = {}
        if target_col and f'importance_{target_col}' in self.analyzer.analysis_results:
            importance_df = self.analyzer.analysis_results[f'importance_{target_col}']
            importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
        
        # Get missing values information
        missing_dict = {}
        if 'missing_values' in self.analyzer.analysis_results:
            missing_df = self.analyzer.analysis_results['missing_values']
            missing_dict = dict(zip(missing_df['feature'], missing_df['missing_pct']))
        
        # Process each correlated group
        for i, group in enumerate(correlated_groups):
            group_name = f"group_{i+1}"
            
            # Skip empty groups
            if not group:
                continue
                
            # Calculate scores for features in this group
            feature_scores = []
            for feature in group:
                # Default score components
                importance_score = importance_dict.get(feature, 0)
                missing_pct = missing_dict.get(feature, 0)
                name_length = len(feature)  # Prefer shorter names as they're often more fundamental
                
                # Calculate overall score (higher is better)
                score = importance_score * (100 - missing_pct) / 100
                
                # Apply small penalty for very long names
                score = score * (1 - (name_length / 1000))
                
                feature_scores.append((feature, score))
            
            # Sort by score and select top feature
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top 1-2 features depending on group size
            num_to_select = min(2, max(1, len(group) // 10))
            selected = [item[0] for item in feature_scores[:num_to_select]]
            
            representatives[group_name] = selected
        
        self.group_representatives = representatives
        return representatives
    
    def create_target_specific_features(self, features_df, target_df=None, methods=None):
        """
        Create target-specific feature sets optimized for different prediction targets.
        
        Args:
            features_df: DataFrame with features
            target_df: Optional DataFrame with targets
            methods: Dict mapping target names to selection methods
            
        Returns:
            Dict mapping target names to feature lists
        """
        if methods is None:
            methods = {
                'win': 'model_based',
                'cut': 'model_based',
                'top3': 'model_based',
                'top10': 'model_based'
            }
        
        # Create target-specific feature sets
        for target_name, method in methods.items():
            target_col = self._map_target_name_to_column(target_name)
            
            if target_df is not None and target_col in target_df.columns:
                # Get features specific to this target
                merged_df = pd.merge(
                    features_df, 
                    target_df[['player_id', 'tournament_id', target_col]],
                    on=['player_id', 'tournament_id'],
                    how='inner'
                )
                
                if method == 'model_based':
                    features = self._select_by_model_importance(
                        merged_df, target_col, n_features=50
                    )
                else:
                    features = self._select_by_correlation(
                        merged_df, target_col, n_features=50
                    )
                
                # Store the selected features
                self.target_specific_features[target_name] = set(features)
        
        return self.target_specific_features
    
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
    
    def _select_by_model_importance(self, df, target_col, n_features=50):
        """Select features using gradient boosting importance."""
        # Get numeric features only
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target and ID columns
        features = [col for col in numeric_cols 
                if col != target_col and col not in ['player_id', 'tournament_id']]
        
        if not features:
            print(f"Warning: No numeric features found for model importance analysis")
            return []
        
        # Prepare data
        X = df[features].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].mean() if not X[col].isna().all() else 0)
            
        y = df[target_col]
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Get feature importances
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Return top N features
        return importance.head(n_features)['feature'].tolist()
    
    def _select_by_correlation(self, df, target_col, n_features=50):
        """Select features using correlation with target."""
        # Get numeric features only
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target and ID columns
        features = [col for col in numeric_cols 
                   if col != target_col and col not in ['player_id', 'tournament_id']]
        
        # Calculate correlation with target
        correlations = []
        
        for feature in features:
            if not df[feature].isna().all():
                corr = df[feature].corr(df[target_col])
                if not pd.isna(corr):
                    correlations.append((feature, abs(corr)))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N features
        return [item[0] for item in correlations[:n_features]]
    
    def create_interaction_features(self, df, base_features=None, target_col=None):
        """
        Create interaction features based on domain knowledge and correlation patterns.
        
        Args:
            df: DataFrame with features
            base_features: Optional list of features to use for interactions
            target_col: Optional target column to evaluate interactions
            
        Returns:
            DataFrame with added interaction features
        """
        if base_features is None:
            base_features = list(self.core_features)
        
        result_df = df.copy()
        interactions = {}
        
        # Define potential interaction groups
        interaction_groups = {
            'form_history': ['recent_', 'history_', 'career_'],
            'skill_course': ['putting_', 'approach_', 'off_tee_', 'course_', 'fit_'],
            'scoring_weather': ['scoring_', 'wind_', 'precip_', 'temp_'],
            'consistency_difficulty': ['consistency_', 'variability_', 'difficulty_']
        }
        
        # Create interactions between groups
        for group_name, patterns in interaction_groups.items():
            # Find features matching each pattern
            group_features = set()
            for pattern in patterns:
                matches = [f for f in base_features if pattern in f.lower()]
                group_features.update(matches)
                
            # Skip if not enough features found
            if len(group_features) < 2:
                continue
                
            # Select top features from group (limit to 3 per group)
            top_group_features = list(group_features)[:3]
            
            # Create interactions with other groups
            for other_group, other_patterns in interaction_groups.items():
                if other_group == group_name:
                    continue
                    
                # Find features in other group
                other_features = set()
                for pattern in other_patterns:
                    matches = [f for f in base_features if pattern in f.lower()]
                    other_features.update(matches)
                    
                # Skip if not enough features found
                if len(other_features) < 2:
                    continue
                    
                # Select top features from other group
                top_other_features = list(other_features)[:3]
                
                # Create pairwise interactions
                for feat1 in top_group_features:
                    for feat2 in top_other_features:
                        # Skip if either feature has too many missing values
                        missing1 = df[feat1].isna().mean()
                        missing2 = df[feat2].isna().mean()
                        
                        if missing1 > 0.3 or missing2 > 0.3:
                            continue
                            
                        # Create interaction name
                        interaction_name = f"interaction_{feat1}_{feat2}"
                        
                        # Create ratio interaction if makes sense
                        try:
                            # Only create ratio if values are mostly positive
                            if (df[feat1] > 0).mean() > 0.9 and (df[feat2] > 0).mean() > 0.9:
                                ratio_name = f"ratio_{feat1}_to_{feat2}"
                                result_df[ratio_name] = df[feat1] / df[feat2].replace(0, np.nan)
                                interactions[ratio_name] = (feat1, feat2, 'ratio')
                        except Exception as e:
                            pass
                        
                        # Create product interaction
                        try:
                            result_df[interaction_name] = df[feat1] * df[feat2]
                            interactions[interaction_name] = (feat1, feat2, 'product')
                        except Exception as e:
                            pass
        
        self.interaction_features = interactions
        return result_df
    
    def evaluate_feature_stability(self, df, target_col, feature_subset=None, 
                              n_splits=5, output_dir=None):
        """
        Evaluate stability of feature importance across time-based folds.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            feature_subset: Optional list of features to evaluate
            n_splits: Number of time-based folds
            output_dir: Directory to save visualization
            
        Returns:
            DataFrame with stability scores for features
        """
        if feature_subset is None:
            feature_subset = list(self.core_features)
        
        # Ensure target column is present
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Get subset of features that exist in dataframe and are numeric
        features = [f for f in feature_subset if f in df.columns 
                and np.issubdtype(df[f].dtype, np.number)]
        
        if not features:
            print("Warning: No valid numeric features found for stability analysis")
            return pd.DataFrame(columns=['feature', 'mean_importance', 'std_importance', 
                                        'min_importance', 'max_importance', 'stability_score'])
        
        # Create a time-based split (assuming data is ordered by time)
        n_samples = len(df)
        fold_size = n_samples // n_splits
        
        importance_results = []
        
        for fold in range(n_splits):
            try:
                # Select fold data
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < n_splits - 1 else n_samples
                
                fold_df = df.iloc[start_idx:end_idx].copy()
                
                # Filter out rows with NaN in target
                valid_idx = fold_df[target_col].notna()
                if valid_idx.sum() == 0:
                    print(f"Warning: Fold {fold} has no valid target values, skipping")
                    continue
                    
                fold_df = fold_df[valid_idx]
                
                # Handle missing values safely in features
                X = fold_df[features].copy()
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(X[col].mean() if not X[col].isna().all() else 0)
                
                y = fold_df[target_col].values  # Convert to numpy array
                
                # Verify we have sufficient data
                if len(y) < 10:
                    print(f"Warning: Fold {fold} has insufficient data ({len(y)} rows), skipping")
                    continue
                
                # Train model
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Store importances for this fold
                fold_importance = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_,
                    'fold': fold
                })
                
                importance_results.append(fold_importance)
            except Exception as e:
                print(f"Warning: Error processing fold {fold}: {str(e)}")
        
        # Check if we have any valid results
        if not importance_results:
            print("No valid importance results across any folds")
            return pd.DataFrame(columns=['feature', 'mean_importance', 'std_importance', 
                                        'min_importance', 'max_importance', 'stability_score'])
        
        # Combine all results
        all_importances = pd.concat(importance_results)
        
        # Calculate stability metrics
        stability = all_importances.groupby('feature').agg({
            'importance': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Rename columns
        stability.columns = ['feature', 'mean_importance', 'std_importance', 
                        'min_importance', 'max_importance']
        
        # Add coefficient of variation (lower is more stable)
        stability['stability_score'] = stability['std_importance'] / stability['mean_importance']
        stability['stability_score'] = stability['stability_score'].fillna(1.0)  # Handle div by zero
        
        # Sort by mean importance
        stability = stability.sort_values('mean_importance', ascending=False)
        
        # Visualize if output directory provided
        if output_dir:
            self._visualize_feature_stability(all_importances, stability, output_dir)
        
        return stability
    
    def _visualize_feature_stability(self, all_importances, stability, output_dir):
        """Create visualizations for feature stability analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot feature importance distribution across folds
        plt.figure(figsize=(12, 10))
        
        # Select top 20 features by mean importance
        top_features = stability.head(20)['feature'].tolist()
        top_importances = all_importances[all_importances['feature'].isin(top_features)]
        
        # Create box plot
        sns.boxplot(x='importance', y='feature', data=top_importances,
                  order=top_features)
        plt.title('Feature Importance Stability Across Time Folds')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_stability_boxplot.png'), dpi=300)
        
        # Create stability score plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='stability_score', y='feature', data=top_features)
        plt.title('Feature Stability Scores (Lower is Better)')
        plt.xlabel('Stability Score (CoV)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_stability_scores.png'), dpi=300)
    
    def get_optimized_feature_set(self, target_type='win', include_interactions=True, 
                                 max_features=50):
        """
        Get the optimized feature set for a specific prediction target.
        
        Args:
            target_type: Target type ('win', 'cut', 'top3', 'top10')
            include_interactions: Whether to include interaction features
            max_features: Maximum number of features to include
            
        Returns:
            List of optimized feature names
        """
        # Start with core features
        selected = set(self.core_features)
        
        # Add target-specific features
        if target_type in self.target_specific_features:
            selected.update(self.target_specific_features[target_type])
        
        # Add interaction features if requested
        if include_interactions and self.interaction_features:
            # Include only interactions of selected features
            for interaction, (feat1, feat2, _) in self.interaction_features.items():
                if feat1 in selected and feat2 in selected:
                    selected.add(interaction)
        
        # Ensure we don't exceed max features
        if len(selected) > max_features:
            # Prioritize using target-specific importance if available
            if self.target_specific_features.get(target_type):
                # Get top features from target-specific set
                target_features = list(self.target_specific_features[target_type])
                
                # Add core features on top
                prioritized = list(self.core_features) + [f for f in target_features 
                                                        if f not in self.core_features]
                
                # Take top max_features
                selected = set(prioritized[:max_features])
            else:
                # Just take first max_features
                selected = set(list(selected)[:max_features])
        
        # Always include player_id and tournament_id
        selected.update(['player_id', 'tournament_id'])
        
        # Convert to list and return
        return list(selected)
    
    def save_feature_sets(self, output_path):
        """
        Save the optimized feature sets to a JSON file.
        
        Args:
            output_path: Path to save the feature sets
            
        Returns:
            None
        """
        import json
        
        # Create dictionary of feature sets
        feature_sets = {
            'core_features': list(self.core_features),
            'group_representatives': self.group_representatives,
            'target_specific_features': {k: list(v) for k, v in self.target_specific_features.items()},
            'interaction_features': {k: {'feature1': v[0], 'feature2': v[1], 'type': v[2]} 
                                  for k, v in self.interaction_features.items()}
        }
        
        # Add optimized sets for each target
        feature_sets['optimized_sets'] = {
            'win': self.get_optimized_feature_set('win'),
            'cut': self.get_optimized_feature_set('cut'),
            'top3': self.get_optimized_feature_set('top3'),
            'top10': self.get_optimized_feature_set('top10')
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(feature_sets, f, indent=2)