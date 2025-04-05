# feature_engineering/feature_selection/feature_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression, chi2
from sklearn.preprocessing import StandardScaler

class FeatureAnalyzer:
    """
    Analyzes features to support feature selection and transformation decisions.
    
    This class provides methods to examine feature properties, correlations,
    distributions, and importance metrics to guide feature selection.
    """
    
    def __init__(self, features_df, target_df=None):
        """
        Initialize the feature analyzer with a features dataframe.
        
        Args:
            features_df: DataFrame containing all features
            target_df: Optional DataFrame containing target variables
        """
        self.features_df = features_df.copy()
        self.target_df = target_df.copy() if target_df is not None else None
        self.player_ids = self.features_df['player_id'].unique() if 'player_id' in self.features_df.columns else []
        self.analysis_results = {}
        
    def analyze_features(self):
        """
        Perform comprehensive analysis of features and store results.
        
        Returns:
            Dict with analysis results
        """
        self.analyze_basic_stats()
        self.analyze_correlations()
        self.analyze_missing_values()
        if self.target_df is not None:
            self.analyze_feature_importance()
        self.analyze_feature_groups()
        
        return self.analysis_results
    
    def analyze_basic_stats(self):
        """
        Analyze basic statistics for each feature.
        """
        # Get numeric columns only
        numeric_features = self.features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Calculate statistics
        stats = self.features_df[numeric_features].describe().T
        stats['variance'] = self.features_df[numeric_features].var()
        stats['missing'] = self.features_df[numeric_features].isna().sum()
        stats['missing_pct'] = (self.features_df[numeric_features].isna().sum() / len(self.features_df)) * 100
        
        # Add low variance flag
        stats['low_variance'] = stats['variance'] < 0.01
        
        # Store results
        self.analysis_results['basic_stats'] = stats
        
        # Identify constant or near-constant features
        low_variance_features = stats.loc[stats['low_variance'], :].index.tolist()
        self.analysis_results['low_variance_features'] = low_variance_features
        
        return stats
    
    def analyze_correlations(self, method='pearson', threshold=0.8):
        """
        Analyze correlations between features.
        
        Args:
            method: Correlation method ('pearson' or 'spearman')
            threshold: Threshold to identify high correlations
            
        Returns:
            DataFrame with feature correlations
        """
        # Get numeric columns only
        numeric_features = self.features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Filter out special columns
        excluded_cols = ['player_id', 'tournament_id', 'feature_year']
        numeric_features = [col for col in numeric_features if col not in excluded_cols]
        
        # Calculate correlation matrix
        corr_matrix = self.features_df[numeric_features].corr(method=method)
        
        # Identify highly correlated feature pairs
        high_corr_pairs = []
        
        # Get upper triangle of correlation matrix
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find feature pairs with correlation above threshold
        for col in upper_tri.columns:
            for idx, value in upper_tri[col].items():
                if abs(value) > threshold:
                    high_corr_pairs.append((idx, col, value))
        
        # Store results
        self.analysis_results['correlation_matrix'] = corr_matrix
        self.analysis_results['high_correlation_pairs'] = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
        
        # Create sets of correlated features for possible elimination
        correlated_groups = self._group_correlated_features(high_corr_pairs, threshold)
        self.analysis_results['correlated_feature_groups'] = correlated_groups
        
        return corr_matrix
    
    def analyze_missing_values(self):
        """
        Analyze missing values across features.
        
        Returns:
            DataFrame with missing value analysis
        """
        # Calculate missing values
        missing = self.features_df.isna().sum().reset_index()
        missing.columns = ['feature', 'missing_count']
        missing['missing_pct'] = (missing['missing_count'] / len(self.features_df)) * 100
        
        # Sort by missing percentage
        missing = missing.sort_values('missing_pct', ascending=False)
        
        # Identify features with high missing values
        high_missing = missing[missing['missing_pct'] > 50]['feature'].tolist()
        
        # Store results
        self.analysis_results['missing_values'] = missing
        self.analysis_results['high_missing_features'] = high_missing
        
        # Special analysis for LIV players
        if len(self.player_ids) > 0:
            missing_by_player = pd.DataFrame({'player_id': self.player_ids})
            
            # Calculate missing percentage for each player
            for player_id in self.player_ids:
                player_data = self.features_df[self.features_df['player_id'] == player_id]
                if not player_data.empty:
                    missing_by_player.loc[missing_by_player['player_id'] == player_id, 'missing_pct'] = \
                        player_data.isna().sum().sum() / (len(player_data.columns) * len(player_data)) * 100
            
            # Identify potential LIV/retired players (very high missing data)
            potential_special_players = missing_by_player[missing_by_player['missing_pct'] > 70]['player_id'].tolist()
            self.analysis_results['potential_special_players'] = potential_special_players
        
        return missing
    
    def analyze_feature_importance(self, target_column='position', method='correlation'):
        """
        Analyze feature importance with respect to target variables.
        
        Args:
            target_column: Target column to analyze against
            method: Method to calculate importance ('correlation' or 'mutual_info')
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.target_df is None:
            return None
        
        # Merge features with target for importance calculation
        merged_df = pd.merge(
            self.features_df,
            self.target_df[['player_id', 'tournament_id', target_column]],
            on=['player_id', 'tournament_id'],
            how='inner'
        )
        
        if merged_df.empty or target_column not in merged_df.columns:
            return None
        
        # Get numeric features
        numeric_features = merged_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [col for col in numeric_features 
                            if col not in ['player_id', 'tournament_id', target_column]]
        
        # Calculate importance based on method
        importance = []
        
        if method == 'correlation':
            for feature in numeric_features:
                if merged_df[feature].notna().sum() > 5:  # Need minimum sample size
                    corr, p_value = pearsonr(
                        merged_df[feature].fillna(merged_df[feature].mean()),
                        merged_df[target_column]
                    )
                    importance.append({
                        'feature': feature,
                        'importance': abs(corr),
                        'sign': np.sign(corr),
                        'p_value': p_value
                    })
                    
        elif method == 'mutual_info':
            # Scale features for MI calculation
            X = merged_df[numeric_features].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X_scaled, merged_df[target_column])
            
            for idx, feature in enumerate(numeric_features):
                importance.append({
                    'feature': feature,
                    'importance': mi_scores[idx],
                    'sign': None,  # MI doesn't provide direction
                    'p_value': None
                })
        
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame(importance)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Store results
        self.analysis_results[f'importance_{target_column}'] = importance_df
        
        # Get top 20% important features
        top_features = importance_df.head(int(len(importance_df) * 0.2))['feature'].tolist()
        self.analysis_results[f'top_features_{target_column}'] = top_features
        
        return importance_df
    
    def analyze_feature_groups(self):
        """
        Analyze features by logical groups to guide selection decisions.
        """
        # Define logical feature groups based on prefixes/keywords
        feature_groups = {
            'player_performance': ['sg_', 'strokes_gained_', 'scoring_', 'driving_', 'putting_'],
            'course_characteristics': ['course_', 'par3_', 'par4_', 'par5_', 'yards_', 'difficulty'],
            'tournament_history': ['avg_finish', 'best_finish', 'appearances', 'cuts_made_pct', 'history_'],
            'recent_form': ['recent_', 'weighted_', 'momentum_', 'trend_'],
            'consistency': ['std', 'consistency_', 'variability', 'position_std'],
            'career': ['career_', 'experience_', 'official_money', 'achievement_'],
            'weather': ['temp', 'humidity', 'windspeed', 'precip'],
            'meta_features': ['component', 'likelihood', 'potential', 'percentile']
        }
        
        # Categorize features into groups
        grouped_features = {}
        all_columns = self.features_df.columns.tolist()
        
        for group_name, patterns in feature_groups.items():
            group_features = []
            for pattern in patterns:
                matches = [col for col in all_columns if pattern in col.lower()]
                group_features.extend(matches)
            
            grouped_features[group_name] = sorted(list(set(group_features)))
        
        # Add uncategorized features
        categorized = [feature for features in grouped_features.values() for feature in features]
        exclude_cols = ['player_id', 'tournament_id', 'feature_year', 'has_']
        uncategorized = [col for col in all_columns 
                         if col not in categorized 
                         and not any(excl in col for excl in exclude_cols)]
        
        grouped_features['uncategorized'] = uncategorized
        
        # Store results
        self.analysis_results['feature_groups'] = grouped_features
        
        # Calculate statistics for each group
        group_stats = {}
        for group, features in grouped_features.items():
            numeric_features = [f for f in features 
                               if f in self.features_df.select_dtypes(include=['int64', 'float64']).columns]
            
            if numeric_features:
                missing_pct = self.features_df[numeric_features].isna().mean().mean() * 100
                group_stats[group] = {
                    'feature_count': len(features),
                    'numeric_count': len(numeric_features),
                    'missing_pct': missing_pct
                }
        
        self.analysis_results['feature_group_stats'] = group_stats
        
        return grouped_features
    
    def _group_correlated_features(self, corr_pairs, threshold):
        """
        Group highly correlated features into clusters.
        
        Args:
            corr_pairs: List of correlated feature pairs (feat1, feat2, corr_value)
            threshold: Correlation threshold
            
        Returns:
            List of feature groups
        """
        # Create a dictionary of correlated features
        corr_dict = {}
        
        for feat1, feat2, corr in corr_pairs:
            if abs(corr) > threshold:
                if feat1 not in corr_dict:
                    corr_dict[feat1] = set()
                if feat2 not in corr_dict:
                    corr_dict[feat2] = set()
                    
                corr_dict[feat1].add(feat2)
                corr_dict[feat2].add(feat1)
        
        # Find groups using a simple clustering approach
        visited = set()
        groups = []
        
        for feature in corr_dict:
            if feature not in visited:
                group = set()
                self._dfs_correlated(feature, corr_dict, visited, group)
                if len(group) > 1:
                    groups.append(sorted(list(group)))
        
        return groups
    
    def _dfs_correlated(self, feature, corr_dict, visited, group):
        """
        Depth-first search to find correlated feature groups.
        """
        visited.add(feature)
        group.add(feature)
        
        if feature in corr_dict:
            for correlated in corr_dict[feature]:
                if correlated not in visited:
                    self._dfs_correlated(correlated, corr_dict, visited, group)
    
    def generate_analysis_report(self, output_path=None):
        """
        Generate a comprehensive analysis report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            String with the analysis report
        """
        if not self.analysis_results:
            self.analyze_features()
        
        report = []
        report.append("# Feature Analysis Report")
        report.append(f"- Total features: {len(self.features_df.columns)}")
        report.append(f"- Total samples: {len(self.features_df)}")
        report.append("")
        
        # Add basic stats summary
        report.append("## Basic Statistics Summary")
        if 'basic_stats' in self.analysis_results:
            low_var = len(self.analysis_results.get('low_variance_features', []))
            report.append(f"- Features with low variance: {low_var}")
        report.append("")
        
        # Add correlation summary
        report.append("## Feature Correlation Summary")
        if 'high_correlation_pairs' in self.analysis_results:
            high_corr = len(self.analysis_results['high_correlation_pairs'])
            report.append(f"- Highly correlated feature pairs: {high_corr}")
            
            if high_corr > 0:
                report.append("\n### Top 10 Highly Correlated Pairs:")
                for feat1, feat2, corr in self.analysis_results['high_correlation_pairs'][:10]:
                    report.append(f"- {feat1} ↔ {feat2}: {corr:.3f}")
        report.append("")
        
        # Add missing values summary
        report.append("## Missing Values Summary")
        if 'missing_values' in self.analysis_results:
            high_missing = len(self.analysis_results.get('high_missing_features', []))
            report.append(f"- Features with >50% missing values: {high_missing}")
            
            if 'potential_special_players' in self.analysis_results:
                special = len(self.analysis_results['potential_special_players'])
                report.append(f"- Potential special case players (e.g., LIV/retired): {special}")
            
            if high_missing > 0:
                report.append("\n### Top 10 Features with Most Missing Values:")
                missing_df = self.analysis_results['missing_values']
                for _, row in missing_df.head(10).iterrows():
                    report.append(f"- {row['feature']}: {row['missing_pct']:.1f}%")
        report.append("")
        
        # Add feature importance summary
        report.append("## Feature Importance Summary")
        for key, value in self.analysis_results.items():
            if key.startswith('importance_'):
                target = key.replace('importance_', '')
                report.append(f"\n### Top 10 Important Features for '{target}':")
                for _, row in value.head(10).iterrows():
                    sign_str = "+" if row['sign'] > 0 else "-" if row['sign'] < 0 else ""
                    report.append(f"- {row['feature']}: {sign_str}{row['importance']:.3f}")
        report.append("")
        
        # Add feature group summary
        report.append("## Feature Group Summary")
        if 'feature_group_stats' in self.analysis_results:
            report.append("\n| Group | Feature Count | Missing % |")
            report.append("| ----- | ------------- | --------- |")
            
            for group, stats in self.analysis_results['feature_group_stats'].items():
                report.append(f"| {group} | {stats['feature_count']} | {stats['missing_pct']:.1f}% |")
        report.append("")
        
        # Add recommendations
        report.append("## Feature Selection Recommendations")
        
        # Low variance features
        if 'low_variance_features' in self.analysis_results and self.analysis_results['low_variance_features']:
            report.append("\n### Consider Removing Low Variance Features:")
            for feat in self.analysis_results['low_variance_features'][:10]:
                report.append(f"- {feat}")
            if len(self.analysis_results['low_variance_features']) > 10:
                report.append(f"- ... and {len(self.analysis_results['low_variance_features']) - 10} more")
        
        # Highly correlated features
        if 'correlated_feature_groups' in self.analysis_results and self.analysis_results['correlated_feature_groups']:
            report.append("\n### Consider Removing Redundant Features from Correlated Groups:")
            for i, group in enumerate(self.analysis_results['correlated_feature_groups'][:5]):
                report.append(f"\nGroup {i+1}:")
                for feat in group:
                    report.append(f"- {feat}")
            if len(self.analysis_results['correlated_feature_groups']) > 5:
                report.append(f"- ... and {len(self.analysis_results['correlated_feature_groups']) - 5} more groups")
        
        # Features with high missing values
        if 'high_missing_features' in self.analysis_results and self.analysis_results['high_missing_features']:
            report.append("\n### Consider Special Handling for Features with High Missing Values:")
            for feat in self.analysis_results['high_missing_features'][:10]:
                report.append(f"- {feat}")
            if len(self.analysis_results['high_missing_features']) > 10:
                report.append(f"- ... and {len(self.analysis_results['high_missing_features']) - 10} more")
        
        # Special case players
        if 'potential_special_players' in self.analysis_results and self.analysis_results['potential_special_players']:
            report.append("\n### Special Case Players Requiring Custom Handling:")
            for player in self.analysis_results['potential_special_players']:
                report.append(f"- Player ID: {player}")
        
        # Add most important features
        report.append("\n### Most Important Features to Retain:")
        for key, value in self.analysis_results.items():
            if key.startswith('top_features_'):
                target = key.replace('top_features_', '')
                report.append(f"\nFor '{target}':")
                for feat in value[:10]:
                    report.append(f"- {feat}")
                if len(value) > 10:
                    report.append(f"- ... and {len(value) - 10} more")
        
        # Compile report
        report_text = "\n".join(report)
        
        # Save report if output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:  # Add encoding='utf-8' here
                f.write(report_text)
        
        return report_text
    
    def plot_correlation_matrix(self, n_features=30, figsize=(15, 12), save_path=None):
        """
        Plot correlation matrix for the top n features.
        
        Args:
            n_features: Number of features to include
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Check if correlation matrix exists
        if 'correlation_matrix' not in self.analysis_results:
            self.analyze_correlations()
        
        # Get correlation matrix and select top features by variance
        corr = self.analysis_results['correlation_matrix']
        
        # Select features with highest variance
        variances = self.features_df[corr.columns].var().sort_values(ascending=False)
        top_features = variances.index[:n_features].tolist()
        
        # Create figure
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr.loc[top_features, top_features], dtype=bool))
        sns.heatmap(
            corr.loc[top_features, top_features],
            mask=mask,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            annot=False,
            square=True,
            linewidths=.5,
            cbar_kws={'shrink': .5}
        )
        plt.title(f'Feature Correlation Matrix (Top {n_features} Features by Variance)')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_missing_values(self, top_n=30, figsize=(12, 8), save_path=None):
        """
        Plot missing values for top features with missing data.
        
        Args:
            top_n: Number of features to include
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Check if missing values analysis exists
        if 'missing_values' not in self.analysis_results:
            self.analyze_missing_values()
        
        # Get missing values data
        missing = self.analysis_results['missing_values']
        
        # Select top features with missing values
        top_missing = missing.head(top_n)
        
        # Create figure
        plt.figure(figsize=figsize)
        sns.barplot(x='missing_pct', y='feature', data=top_missing)
        plt.title(f'Top {top_n} Features with Missing Values')
        plt.xlabel('Missing Values (%)')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_feature_importance(self, target_column='position', top_n=30, figsize=(12, 8), save_path=None):
        """
        Plot feature importance for a target variable.
        
        Args:
            target_column: Target column for importance calculation
            top_n: Number of features to include
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Check if feature importance analysis exists
        importance_key = f'importance_{target_column}'
        if importance_key not in self.analysis_results:
            self.analyze_feature_importance(target_column=target_column)
        
        # Get feature importance data
        importance = self.analysis_results[importance_key]
        
        # Select top important features
        top_importance = importance.head(top_n)
        
        # Create figure
        plt.figure(figsize=figsize)
        bars = sns.barplot(x='importance', y='feature', data=top_importance)
        
        # Color bars by sign if available
        if 'sign' in top_importance.columns and not top_importance['sign'].isna().all():
            for i, bar in enumerate(bars.patches):
                sign = top_importance.iloc[i]['sign']
                if sign > 0:
                    bar.set_color('green')
                elif sign < 0:
                    bar.set_color('red')
        
        plt.title(f'Top {top_n} Features by Importance for {target_column}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def get_feature_recommendations(self):
        """
        Get feature selection recommendations based on analysis.
        
        Returns:
            Dict with recommended features to keep or remove
        """
        if not self.analysis_results:
            self.analyze_features()
        
        recommendations = {
            'remove': {
                'low_variance': self.analysis_results.get('low_variance_features', []),
                'high_missing': self.analysis_results.get('high_missing_features', [])
            },
            'keep': {},
            'redundant_groups': self.analysis_results.get('correlated_feature_groups', []),
            'special_cases': {
                'players': self.analysis_results.get('potential_special_players', [])
            }
        }
        
        # Add important features to keep
        for key, value in self.analysis_results.items():
            if key.startswith('top_features_'):
                target = key.replace('top_features_', '')
                recommendations['keep'][target] = value
        
        return recommendations