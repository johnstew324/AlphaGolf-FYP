# feature_engineering/feature_selection/feature_selector.py

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

class FeatureSelector:
    """
    Implements feature selection strategies for golf tournament prediction.
    
    This class provides methods to select features based on various criteria
    including variance, correlation, statistical tests, and model-based importance.
    """
    
    def __init__(self, features_df, analyzer=None, target_df=None):
        """
        Initialize the feature selector with a features dataframe.
        
        Args:
            features_df: DataFrame containing all features
            analyzer: Optional FeatureAnalyzer instance with pre-computed analysis
            target_df: Optional DataFrame containing target variables
        """
        self.features_df = features_df.copy()
        self.analyzer = analyzer
        self.target_df = target_df.copy() if target_df is not None else None
        self.selected_features = []
        self.excluded_features = []
        self.selection_history = []
        
    def select_features(self, method='combined', params=None):
        """
        Select features using the specified method.
        
        Args:
            method: Selection method ('variance', 'correlation', 'importance', 
                   'statistical', 'model_based', or 'combined')
            params: Parameters for the selection method
            
        Returns:
            DataFrame with selected features
        """
        # Use default parameters if not provided
        if params is None:
            params = {}
            
        # Set excluded columns that should never be selected
        exclude_always = ['feature_year', 'tournament_id_standard', 'has_', 'collected_at']
        id_columns = ['player_id', 'tournament_id']
        
        # Keep track of the selection method
        self.selection_history.append({
            'method': method,
            'params': params
        })
        
        # Apply the selected method
        if method == 'variance':
            selected = self._select_by_variance(**params)
        elif method == 'correlation':
            selected = self._select_by_correlation(**params)
        elif method == 'importance':
            selected = self._select_by_importance(**params)
        elif method == 'statistical':
            selected = self._select_by_statistical_tests(**params)
        elif method == 'model_based':
            selected = self._select_by_model_importance(**params)
        elif method == 'combined':
            selected = self._select_combined(**params)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Always include ID columns
        for col in id_columns:
            if col in self.features_df.columns and col not in selected:
                selected.append(col)
        
        # Update excluded features list
        all_features = [col for col in self.features_df.columns 
                       if not any(excl in col for excl in exclude_always)]
        self.excluded_features = [col for col in all_features if col not in selected]
        
        # Update selected features
        self.selected_features = selected
        
        # Return the selected features dataframe
        return self.features_df[self.selected_features]
    
    def _select_by_variance(self, threshold=0.01, exempt_columns=None):
        """
        Select features based on variance threshold.
        
        Args:
            threshold: Variance threshold (features with lower variance will be removed)
            exempt_columns: List of column patterns to exempt from this filtering
            
        Returns:
            List of selected feature names
        """
        # Get numeric columns only
        numeric_features = self.features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove ID columns and other special columns
        id_cols = ['player_id', 'tournament_id']
        numeric_features = [col for col in numeric_features if col not in id_cols]
        
        # Apply exemptions if provided
        if exempt_columns:
            exempt = []
            for pattern in exempt_columns:
                exempt.extend([col for col in numeric_features if pattern in col])
            numeric_features = [col for col in numeric_features if col not in exempt]
        
        # Calculate variance for each feature
        variances = self.features_df[numeric_features].var()
        
        # Select features above the threshold
        selected_numeric = variances[variances >= threshold].index.tolist()
        
        # Add exempt columns back
        if exempt_columns:
            for pattern in exempt_columns:
                selected_numeric.extend([col for col in self.features_df.columns if pattern in col])
        
        # Add non-numeric columns
        categorical_cols = [col for col in self.features_df.columns 
                            if col not in numeric_features and col not in id_cols]
        
        # Combine all selected features
        selected = list(set(selected_numeric + categorical_cols + id_cols))
        
        return selected
    
    def _select_by_correlation(self, threshold=0.85, method='pearson', 
                              target_col=None, prefer_recent=True):
        """
        Select features by removing highly correlated ones.
        
        Args:
            threshold: Correlation threshold (features with higher correlation will be candidates for removal)
            method: Correlation method ('pearson' or 'spearman')
            target_col: If provided, prioritize features more correlated with target
            prefer_recent: If True, prefer recent features over historical ones when correlated
            
        Returns:
            List of selected feature names
        """
        # Get numeric columns only
        numeric_features = self.features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove ID columns
        id_cols = ['player_id', 'tournament_id']
        numeric_features = [col for col in numeric_features if col not in id_cols]
        
        # Calculate correlation matrix
        corr_matrix = self.features_df[numeric_features].corr(method=method)
        
        # Find features to drop
        features_to_drop = []
        
        # If we have a target column, calculate target correlations
        target_corrs = {}
        if target_col and self.target_df is not None:
            # Merge features with target
            merged = pd.merge(
                self.features_df, 
                self.target_df[['player_id', 'tournament_id', target_col]],
                on=['player_id', 'tournament_id'],
                how='inner'
            )
            
            # Calculate correlation with target
            for feature in numeric_features:
                if not merged[feature].isna().all():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        corr = merged[feature].corr(merged[target_col], method=method)
                        target_corrs[feature] = abs(corr) if not pd.isna(corr) else 0
        
        # Helper function to decide which feature to keep when a pair is correlated
        def feature_priority(feat1, feat2):
            # Priority logic rules:
            # 1. If target correlations exist, keep feature with higher target correlation
            if target_col and feat1 in target_corrs and feat2 in target_corrs:
                if abs(target_corrs[feat1] - target_corrs[feat2]) > 0.05:  # Meaningful difference
                    return feat1 if target_corrs[feat1] > target_corrs[feat2] else feat2
            
            # 2. If prefer_recent, check for recent vs historical features
            if prefer_recent:
                recent_patterns = ['recent_', 'momentum_', 'current_', 'trend_']
                historical_patterns = ['history_', 'career_', 'avg_']
                
                feat1_is_recent = any(pattern in feat1 for pattern in recent_patterns)
                feat2_is_recent = any(pattern in feat2 for pattern in recent_patterns)
                feat1_is_historical = any(pattern in feat1 for pattern in historical_patterns)
                feat2_is_historical = any(pattern in feat2 for pattern in historical_patterns)
                
                if feat1_is_recent and feat2_is_historical:
                    return feat1
                elif feat2_is_recent and feat1_is_historical:
                    return feat2
            
            # 3. Default to shorter-named feature (usually simpler/more fundamental)
            return feat1 if len(feat1) <= len(feat2) else feat2
        
        # Iterate through correlation matrix to find highly correlated pairs
        for i in range(len(numeric_features)):
            if numeric_features[i] in features_to_drop:
                continue
                
            for j in range(i+1, len(numeric_features)):
                if numeric_features[j] in features_to_drop:
                    continue
                    
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    # Determine which feature to keep
                    feat1, feat2 = numeric_features[i], numeric_features[j]
                    keep_feature = feature_priority(feat1, feat2)
                    drop_feature = feat2 if keep_feature == feat1 else feat1
                    
                    features_to_drop.append(drop_feature)
        
        # Get selected features (all except dropped)
        selected_numeric = [f for f in numeric_features if f not in features_to_drop]
        
        # Add non-numeric and ID columns
        categorical_cols = [col for col in self.features_df.columns 
                           if col not in numeric_features and col not in id_cols]
        
        # Combine all selected features
        selected = list(set(selected_numeric + categorical_cols + id_cols))
        
        return selected
    
    def _select_by_importance(self, n_features=100, target_col='position', include_profile=True):
        """
        Select most important features based on correlation with target.
        
        Args:
            n_features: Number of features to select
            target_col: Target column name
            include_profile: If True, include demographic/profile features regardless of importance
            
        Returns:
            List of selected feature names
        """
        if self.target_df is None:
            raise ValueError("Target dataframe required for importance-based selection")
        
        # Merge features with target
        merged = pd.merge(
            self.features_df, 
            self.target_df[['player_id', 'tournament_id', target_col]],
            on=['player_id', 'tournament_id'],
            how='inner'
        )
        
        if merged.empty:
            raise ValueError("No matching data between features and targets")
        
        # Get numeric features only
        numeric_features = merged.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [col for col in numeric_features 
                           if col not in ['player_id', 'tournament_id', target_col]]
        
        # Calculate importance based on correlation with target
        importance = []
        
        for feature in numeric_features:
            if not merged[feature].isna().all():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr = merged[feature].corr(merged[target_col])
                    importance.append({
                        'feature': feature,
                        'importance': abs(corr) if not pd.isna(corr) else 0
                    })
        
        # Sort by importance
        importance_df = pd.DataFrame(importance).sort_values('importance', ascending=False)
        
        # Select top N features
        top_features = importance_df.head(n_features)['feature'].tolist()
        
        # Add ID columns
        id_cols = ['player_id', 'tournament_id']
        
        # Add profile features if requested
        if include_profile:
            profile_cols = [col for col in self.features_df.columns 
                           if any(pat in col for pat in ['name', 'country', 'tour', 'owgr'])]
            top_features.extend(profile_cols)
        
        # Combine all selected features
        selected = list(set(top_features + id_cols))
        
        return selected
    
    def _select_by_statistical_tests(self, n_features=100, target_col='position', method='f_regression'):
        """
        Select features using statistical hypothesis tests.
        
        Args:
            n_features: Number of features to select
            target_col: Target column name
            method: Statistical test method ('f_regression' or 'mutual_info')
            
        Returns:
            List of selected feature names
        """
        if self.target_df is None:
            raise ValueError("Target dataframe required for statistical selection")
        
        # Merge features with target
        merged = pd.merge(
            self.features_df, 
            self.target_df[['player_id', 'tournament_id', target_col]],
            on=['player_id', 'tournament_id'],
            how='inner'
        )
        
        if merged.empty:
            raise ValueError("No matching data between features and targets")
        
        # Get numeric features only
        numeric_features = merged.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [col for col in numeric_features 
                           if col not in ['player_id', 'tournament_id', target_col]]
        
        # Prepare feature matrix and target vector
        X = merged[numeric_features].fillna(0)
        y = merged[target_col]
        
        # Select features using statistical test
        if method == 'f_regression':
            selector = SelectKBest(f_regression, k=min(n_features, len(numeric_features)))
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=min(n_features, len(numeric_features)))
        else:
            raise ValueError(f"Unknown statistical test method: {method}")
        
        # Apply selection
        selector.fit(X, y)
        
        # Get mask of selected features
        selected_mask = selector.get_support()
        
        # Get selected feature names
        selected_numeric = [numeric_features[i] for i in range(len(numeric_features)) if selected_mask[i]]
        
        # Add non-numeric and ID columns
        id_cols = ['player_id', 'tournament_id']
        categorical_cols = [col for col in self.features_df.columns 
                            if col not in numeric_features and col not in id_cols]
        
        # Combine all selected features
        selected = list(set(selected_numeric + categorical_cols + id_cols))
        
        return selected
    
    def _select_by_model_importance(self, n_features=100, target_col='position', model='rf'):
        """
        Select features using model-based importance.
        
        Args:
            n_features: Number of features to select
            target_col: Target column name
            model: Model to use for importance calculation ('rf' or 'gbm')
            
        Returns:
            List of selected feature names
        """
        if self.target_df is None:
            raise ValueError("Target dataframe required for model-based selection")
        
        # Merge features with target
        merged = pd.merge(
            self.features_df, 
            self.target_df[['player_id', 'tournament_id', target_col]],
            on=['player_id', 'tournament_id'],
            how='inner'
        )
        
        if merged.empty:
            raise ValueError("No matching data between features and targets")
        
        # Get numeric features only
        numeric_features = merged.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [col for col in numeric_features 
                           if col not in ['player_id', 'tournament_id', target_col]]
        
        # Prepare feature matrix and target vector
        X = merged[numeric_features].fillna(0)
        y = merged[target_col]
        
        # Train model for feature importance
        if model == 'rf':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model == 'gbm':
            estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model}")
        
        # Fit model and get feature importances
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X, y)
        
        # Get feature importances
        importances = estimator.feature_importances_
        
        # Create DataFrame with feature names and importances
        importance_df = pd.DataFrame({
            'feature': numeric_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        top_features = importance_df.head(n_features)['feature'].tolist()
        
        # Add ID columns and categorical columns
        id_cols = ['player_id', 'tournament_id']
        categorical_cols = [col for col in self.features_df.columns 
                           if col not in numeric_features and col not in id_cols]
        
        # Combine all selected features
        selected = list(set(top_features + categorical_cols + id_cols))
        
        return selected
    
    def _select_combined(self, n_features=100, variance_threshold=0.01, corr_threshold=0.85, 
                     target_col='position', include_special_columns=True):
        """
        Apply a combined feature selection strategy with preference for predictive features.
        
        Args:
            n_features: Target number of features to select
            variance_threshold: Threshold for variance filtering
            corr_threshold: Threshold for correlation filtering
            target_col: Target column for importance-based selection
            include_special_columns: Whether to include special columns regardless of other criteria
            
        Returns:
            List of selected feature names
        """
        # First identify predictive features
        predictive_features = self._identify_predictive_features(self.features_df)
        
        # Start with predictive features
        filtered_features = set(predictive_features)
        
        # Apply variance filtering (on numeric features only)
        variance_selected = set(self._select_by_variance(threshold=variance_threshold))
        
        # Apply correlation filtering (on numeric features only)
        correlation_selected = set(self._select_by_correlation(threshold=corr_threshold, 
                                                            target_col=target_col))
        
        # Prioritize features that pass both filters and are predictive
        filtered_features = filtered_features.intersection(variance_selected).intersection(correlation_selected)
        
        # Special columns that should always be included if present
        if include_special_columns:
            special_columns = ['player_id', 'tournament_id', 'tournament_id_standard']
            filtered_features.update(special_columns)
        
        # Ensure all selected features are in the original dataframe
        selected = [col for col in filtered_features if col in self.features_df.columns]
        
        return selected
    
    def select_by_group(self, group_config, n_features=100):
        """
        Select features by logical groups with specified proportions.
        
        Args:
            group_config: Dict mapping group patterns to selection counts/proportions
            n_features: Total target number of features (used for proportion calculation)
            
        Returns:
            DataFrame with selected features
        """
        # Categorize features into groups
        groups = {}
        for pattern, count in group_config.items():
            # Find features matching the pattern
            matching = [col for col in self.features_df.columns if pattern in col.lower()]
            groups[pattern] = matching
        
        # Calculate counts if proportions provided
        feature_counts = {}
        for pattern, count in group_config.items():
            if isinstance(count, float) and 0 < count < 1:
                # Convert proportion to count
                feature_counts[pattern] = int(count * n_features)
            else:
                # Use absolute count
                feature_counts[pattern] = count
        
        # Select features from each group
        selected = []
        
        for pattern, group_features in groups.items():
            if not group_features:
                continue
                
            count = feature_counts.get(pattern, 0)
            count = min(count, len(group_features))
            
            # For now use variance as selection criterion within groups
            # This could be enhanced to use other metrics
            variances = self.features_df[group_features].var()
            top_features = variances.sort_values(ascending=False).head(count).index.tolist()
            
            selected.extend(top_features)
        
        # Add ID columns
        id_cols = ['player_id', 'tournament_id']
        for col in id_cols:
            if col in self.features_df.columns and col not in selected:
                selected.append(col)
        
        # Update internal state
        self.selected_features = selected
        
        # Return selected features
        return self.features_df[selected]
    
    def handle_special_players(self, special_ids=None, strategy='fallback_features'):
        """
        Handle special case players like LIV golfers or retired players.
        
        Args:
            special_ids: List of player IDs to handle specially (if None, auto-detect)
            strategy: Strategy to handle special players ('fallback_features' or 'imputation')
            
        Returns:
            Updated features DataFrame
        """
        if special_ids is None:
            # Auto-detect special players (high missing values)
            numeric_cols = self.features_df.select_dtypes(include=['int64', 'float64']).columns
            
            missing_counts = {}
            for player_id in self.features_df['player_id'].unique():
                player_data = self.features_df[self.features_df['player_id'] == player_id]
                missing_pct = player_data[numeric_cols].isna().mean().mean()
                missing_counts[player_id] = missing_pct
            
            # Players with over 70% missing values are considered special cases
            special_ids = [pid for pid, pct in missing_counts.items() if pct > 0.7]
        
        # Handle the special players
        if strategy == 'fallback_features':
            # Define essential fallback features that should be available for all players
            fallback_features = ['player_id', 'tournament_id', 'owgr', 'career_wins', 
                                 'career_events', 'career_top10_rate', 'career_cuts_made_rate']
            
            # Check which features are available for special players
            available_features = {}
            
            for player_id in special_ids:
                player_data = self.features_df[self.features_df['player_id'] == player_id]
                if not player_data.empty:
                    for col in fallback_features:
                        if col in player_data.columns and not player_data[col].isna().all():
                            if player_id not in available_features:
                                available_features[player_id] = []
                            available_features[player_id].append(col)
            
            # Create a special flag for these players
            for player_id in special_ids:
                self.features_df.loc[self.features_df['player_id'] == player_id, 'is_special_case'] = 1
            
            # Fill other rows with 0
            if 'is_special_case' in self.features_df.columns:
                self.features_df['is_special_case'] = self.features_df['is_special_case'].fillna(0)
            else:
                self.features_df['is_special_case'] = 0
            
            # Add info about available features for special players
            self.special_player_info = {
                'ids': special_ids,
                'available_features': available_features
            }
            
        elif strategy == 'imputation':
            # Implement imputation strategy for special players
            # For example, using mean/median values or nearest neighbor imputation
            # This would require more complex imputation logic
            pass
        
        return self.features_df
    
    def save_selection_config(self, filepath):
        """
        Save the feature selection configuration to a file.
        
        Args:
            filepath: Path to save the configuration
            
        Returns:
            None
        """
        import json
        
        config = {
            'selected_features': self.selected_features,
            'excluded_features': self.excluded_features,
            'selection_history': self.selection_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_selection_config(self, filepath):
        """
        Load a feature selection configuration from a file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            None
        """
        import json
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.selected_features = config.get('selected_features', [])
        self.excluded_features = config.get('excluded_features', [])
        self.selection_history = config.get('selection_history', [])
    
    def get_selected_features_df(self):
        """
        Get DataFrame with only the selected features.
        
        Returns:
            DataFrame with selected features
        """
        if not self.selected_features:
            raise ValueError("No features have been selected. Call select_features() first.")
        
        return self.features_df[self.selected_features]
    
    
    def filter_non_predictive_features(self, features_df):
        """
        Explicitly filter out non-predictive features like descriptions and IDs.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            DataFrame with filtered features
        """
        # Define patterns to exclude
        exclude_patterns = [
            'overview_', 'snapshot_', 'desc', 'name', 'country', 'state', 'city', 
            'design', 'rough', 'green', 'fairway', 'location'
        ]
        
        # Keep these specific features even if they match patterns
        keep_features = [
            'player_id', 'tournament_id', 'tournament_id_standard',
            'victory_potential', 'owgr', 'strokes_gained_scoring_sg_total'
        ]
        
        # Identify columns to drop
        cols_to_drop = []
        for col in features_df.columns:
            if any(pattern in col.lower() for pattern in exclude_patterns):
                if col not in keep_features:
                    cols_to_drop.append(col)
        
        # Drop non-predictive features
        filtered_df = features_df.drop(columns=cols_to_drop)
        
        # Update selected features
        self.selected_features = [col for col in self.selected_features if col in filtered_df.columns]
        
        return filtered_df

    def _identify_predictive_features(self, df):
        """
        Identify features likely to have predictive power.
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of predictive feature names
        """
        # Start with numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Exclude IDs and special columns
        exclude_ids = ['player_id', 'tournament_id', 'tournament_id_standard']
        numeric_cols = [col for col in numeric_cols if col not in exclude_ids]
        
        # Exclude other non-predictive patterns
        non_predictive_patterns = [
            'collected_at', 'timestamp', 'date', 'logo', 'webview', 'id'
        ]
        
        predictive_features = [col for col in numeric_cols 
                            if not any(pattern in col.lower() for pattern in non_predictive_patterns)]
        
        # Add back essential columns
        predictive_features += exclude_ids
        
        return predictive_features