"""
winners_pipeline.py - A dedicated pipeline to create a winners-specific feature dataset

This pipeline processes the predictive features dataset to create a refined
feature set specifically optimized for predicting tournament winners.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

class WinnersFeaturePipeline:
    """
    Pipeline to process golf tournament predictive features and create a dataset
    focused on predicting tournament winners.
    """
    
    def __init__(self, input_file='feature_analysis/predictive_features.csv', output_dir='output'):
        """
        Initialize the pipeline with file paths.
        
        Args:
            input_file: Path to the predictive features CSV file
            output_dir: Directory where the winner features will be saved
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(output_dir, 'winners_pipeline.log'))
            ]
        )
        self.logger = logging.getLogger('winners_pipeline')
        
        # This is a match for the identified winners who all have this position value
        self.winner_position_value = -0.11215832619799004
        
        # Key features for winner prediction, based on analysis
        self.core_winner_features = [
            # Player identification
            'player_id',
            'tournament_id',
            
            
            
            #needs scaling 
            "avg_total_score_history_interaction",
            "victory_potential",
            'career_success_score',
            "earnings",
            
            'scoring_birdie_average',
            ""
            
            
            # no scaling 
            "scoring_late_scoring_average_rank_scaled",
            'strokes_gained_scoring_sg_total',
            'strokes_gained_putting_sg_putting',
            'strokes_gained_driving_sg_off_the_tee',
            'scoring_birdie_average',
            'career_top_10_pct',
            "owgr_percentile_interaction",
            "scoring_late_scoring_average_rank_scaled",
            
            'scoring_scoring_average_adjusted', 
            'career_success_score',
            "victory_potential",
            "avg_total_score_history_interaction",
            "cuts_made_ratio",
            "career_success_score_percentile_interaction",
            "earnings",
            "recent_trend_slope_scaled",
            
            
            # Secondary predictive features
            'recent_top10_rate',
            'recent_top25_rate',

            'career_wins',
            'recent_wins',
            'recent_best_finish',
            'avg_finish',
            'last3_position_numeric',
            'position_momentum',
            'history_sg_tot_scaled',
            
            # Tournament context features
            'owgr_tier_numeric',
            'experience_level_numeric',
            'form_vs_history',
            "course_history_score",
            "scoring_birdie_average",
            
            # Course fit-related features
            'course_fit_score',
            'course_history_score',
            "owgr_percentile_interaction",
            'owgr_score',
            
            "putting_one_putt_percentage_rank_scaled",
            "course_pars_pct_interaction",
            'course_par5_birdies_interaction',
            'course_par4_birdies_interaction',
            "par4_birdie_pct",
            "par5_eagle_pct",
            "scoring_birdie_average_rank_scaled",
            "scoring_round_4_scoring_average_rank_scaled",
            "par3_bogey_pct",

            
            # Consistency and momentum
            'avg_birdie_streak',
            'score_trend',
            'top10_consistency_scaled',
            'recent_score_std',
            'recent_finish_std',
            'recent_trend_slope_scaled',
            "score_trend_temporal",
            
            # Target variables - will create these if not present
            'wins_current',
            'winner',
            "career_wins_2_scaled",
            
            #weather 
            "wind_range",
            "avg_windgust",
            "career_official_money"
            
        ]
        
        # Additional target variables we'll create
        self.target_variables = ['winner', 'top3', 'top10', 'top25']
        
        # Load data once initialized
        self.data = None
    
    def load_data(self):
        """
        Load the predictive features dataset.
        
        Returns:
            self for method chaining
        """
        self.logger.info(f"Loading data from {self.input_file}")
        try:
            self.data = pd.read_csv(self.input_file)
            self.logger.info(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
            
            # Log some basic info about the dataset
            tournaments = self.data['tournament_id'].unique()
            players = self.data['player_id'].unique()
            self.logger.info(f"Dataset contains {len(tournaments)} tournaments and {len(players)} players")
            
            # Check if wins_current exists
            if 'wins_current' in self.data.columns:
                # Handle NaN values in wins_current by treating them as 0
                self.data['wins_current'] = self.data['wins_current'].fillna(0)
                winner_count = self.data[self.data['wins_current'] > 0].shape[0]
                self.logger.info(f"Found {winner_count} winners in the dataset (based on wins_current)")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.data = pd.DataFrame()
            
        return self
    
    def create_target_variables(self):
        """
        Create target variables for winner prediction.
        
        This creates several targets:
        - winner: Binary target for tournament winners
        - top3: Binary target for top 3 finishers
        - top10: Binary target for top 10 finishers
        - top25: Binary target for top 25 finishers
        
        Returns:
            self for method chaining
        """
        if self.data is None or self.data.empty:
            self.logger.error("No data loaded. Call load_data() first.")
            return self
        
        self.logger.info("Creating target variables")
        
        # Create winner target based on wins_current
        if 'wins_current' in self.data.columns:
            # Fill NaN values with 0 to ensure proper comparison
            self.data['wins_current'] = self.data['wins_current'].fillna(0)
            self.data['winner'] = (self.data['wins_current'] > 0).astype(int)
            self.logger.info(f"Created winner target variable from wins_current")
            
            # Validate that winners have the expected finish position value
            winners = self.data[self.data['winner'] == 1]
            winner_positions = winners['finish_position_std_scaled'].unique()
            self.logger.info(f"Winners have finish position values: {winner_positions}")
            
            # If all winners have the consistent finish position, we can also use this as a cross-check
            if len(winner_positions) == 1:
                self.logger.info(f"All winners have consistent position value: {winner_positions[0] == self.winner_position_value}")
            else:
                self.logger.warning("Winners have inconsistent position values, using wins_current as source of truth")
        else:
            # If wins_current doesn't exist, use finish_position_std_scaled as a proxy
            self.logger.warning("wins_current not found, using finish_position_std_scaled to determine winners")
            self.data['winner'] = (self.data['finish_position_std_scaled'] == self.winner_position_value).astype(int)
        
        # Create additional placement targets
        for position_col in ['last1_position_numeric', 'most_recent_position', 'finish_position', 'avg_finish']:
            if position_col in self.data.columns:
                # Fill NaN values with a high number to avoid comparison issues
                self.data[position_col] = self.data[position_col].fillna(9999)
                self.data['top3'] = (self.data[position_col] <= 3).astype(int)
                self.data['top10'] = (self.data[position_col] <= 10).astype(int)
                self.data['top25'] = (self.data[position_col] <= 25).astype(int)
                self.logger.info(f"Created additional target variables from {position_col}")
                break
        
        # Count how many of each target we have
        for target in self.target_variables:
            if target in self.data.columns:
                count = self.data[target].sum()
                self.logger.info(f"Target {target}: {count} positives ({100*count/len(self.data):.2f}%)")
        
        return self
    
    def select_winner_features(self):
        """
        Select the most important features for winner prediction.
        
        Returns:
            self for method chaining
        """
        if self.data is None or self.data.empty:
            self.logger.error("No data loaded. Call load_data() first.")
            return self
        
        self.logger.info("Selecting winner-specific features")
        
        # Get available columns that are in our feature list
        available_columns = [col for col in self.core_winner_features if col in self.data.columns]
        
        # Also add any target variables we created
        for target in self.target_variables:
            if target in self.data.columns and target not in available_columns:
                available_columns.append(target)
        
        self.logger.info(f"Selected {len(available_columns)} features for the winners dataset")
        
        # Create the winners dataset with selected features
        self.winners_data = self.data[available_columns].copy()
        
        # Log missing features that we couldn't include
        missing_features = set(self.core_winner_features) - set(available_columns)
        if missing_features:
            self.logger.warning(f"Could not include these features (not found in dataset): {missing_features}")
        
        return self
    
    def analyze_features(self):
        """
        Analyze the selected features to understand patterns of winners vs non-winners.
        
        Returns:
            self for method chaining
        """
        if not hasattr(self, 'winners_data') or self.winners_data is None or self.winners_data.empty:
            self.logger.error("No winners data created. Call select_winner_features() first.")
            return self
        
        if 'winner' not in self.winners_data.columns:
            self.logger.error("Winner target not found in dataset.")
            return self
        
        self.logger.info("Analyzing features for winners vs non-winners")
        
        # Get numeric features for analysis (excluding ID columns and target variables)
        exclude_cols = ['player_id', 'tournament_id', 'tournament_id_standard'] + self.target_variables
        numeric_features = [col for col in self.winners_data.columns 
                           if col not in exclude_cols 
                           and pd.api.types.is_numeric_dtype(self.winners_data[col])]
        
        # Initialize a dataframe to store feature comparison results
        comparison_data = []
        
        # Calculate mean values for winners vs non-winners for each feature
        winners_df = self.winners_data[self.winners_data['winner'] == 1]
        non_winners_df = self.winners_data[self.winners_data['winner'] == 0]
        
        for feature in numeric_features:
            # Calculate means, handling empty or all-NaN cases
            winner_mean = winners_df[feature].mean() if not winners_df[feature].isna().all() else np.nan
            non_winner_mean = non_winners_df[feature].mean() if not non_winners_df[feature].isna().all() else np.nan
            
            # Skip if both means are NaN
            if pd.isna(winner_mean) and pd.isna(non_winner_mean):
                continue
                
            # Calculate absolute difference
            abs_diff = abs(winner_mean - non_winner_mean) if not (pd.isna(winner_mean) or pd.isna(non_winner_mean)) else np.nan
            
            # Calculate importance metric (normalized difference)
            avg_value = (abs(winner_mean) + abs(non_winner_mean)) / 2 if not (pd.isna(winner_mean) or pd.isna(non_winner_mean)) else np.nan
            importance = abs_diff / avg_value if avg_value != 0 and not pd.isna(avg_value) else np.nan
            
            # Add to comparison data
            comparison_data.append({
                'feature': feature,
                'winner_mean': winner_mean,
                'non_winner_mean': non_winner_mean,
                'abs_diff': abs_diff,
                'importance': importance
            })
        
        # Convert to DataFrame
        feature_comparison = pd.DataFrame(comparison_data)
        
        # Sort by importance, handling NaN values
        if not feature_comparison.empty:
            feature_comparison = feature_comparison.sort_values('importance', ascending=False, na_position='last')
            
            # Save feature importance analysis
            feature_comparison.to_csv(self.output_dir / 'winner_feature_importance.csv', index=False)
            self.logger.info(f"Saved feature importance analysis to {self.output_dir / 'winner_feature_importance.csv'}")
            
            # Log top features by importance
            top_features = feature_comparison.head(10)
            self.logger.info("Top 10 features by importance:")
            for _, row in top_features.iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f} | Winner: {row['winner_mean']:.4f}, Non-winner: {row['non_winner_mean']:.4f}")
        else:
            self.logger.warning("No valid feature comparison data could be generated")
        
        return self
    
    def save_dataset(self, filename='winners_features.csv'):
        """
        Save the winners feature dataset to CSV.
        
        Args:
            filename: Name of the output file
            
        Returns:
            self for method chaining
        """
        if not hasattr(self, 'winners_data') or self.winners_data is None or self.winners_data.empty:
            self.logger.error("No winners data created. Call select_winner_features() first.")
            return self
        
        output_path = self.output_dir / filename
        self.winners_data.to_csv(output_path, index=False)
        self.logger.info(f"Saved winners feature dataset to {output_path}")
        self.logger.info(f"Dataset shape: {self.winners_data.shape}")
        
        return self
    
    def run(self):
        """
        Run the full pipeline from loading data to saving the winners dataset.
        
        Returns:
            DataFrame with the winners feature dataset
        """
        self.logger.info("Starting winners feature pipeline")
        
        try:
            # Execute each step in the pipeline, handling potential errors
            self.load_data()
            if self.data is None or self.data.empty:
                self.logger.error("Failed to load data. Pipeline cannot continue.")
                return None
                
            self.create_target_variables()
            self.select_winner_features()
            
            # These steps might fail if data is incompatible
            try:
                self.analyze_features()
            except Exception as e:
                self.logger.warning(f"Feature analysis step failed: {str(e)}")
                self.logger.warning("Continuing with pipeline despite analysis failure")
            
            self.save_dataset()
            
            self.logger.info("Winners feature pipeline completed successfully")
            return self.winners_data
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


def main():
    """Run the winners feature pipeline as a standalone script."""
    # Default path to predictive features
    input_file = 'feature_analysis/predictive_features.csv'
    
    # Check if file exists, otherwise try alternative locations
    if not os.path.exists(input_file):
        alternatives = [
            'predictive_features.csv',
            '../feature_analysis/predictive_features.csv',
            '../predictive_features.csv'
        ]
        
        for alt in alternatives:
            if os.path.exists(alt):
                input_file = alt
                break
    
    # Create and run the pipeline
    pipeline = WinnersFeaturePipeline(input_file=input_file)
    winners_data = pipeline.run()
    
    # Print some stats about the dataset
    if winners_data is not None and not winners_data.empty:
        print(f"\nWinners dataset created with {winners_data.shape[0]} rows and {winners_data.shape[1]} columns")
        
        if 'winner' in winners_data.columns:
            winner_count = winners_data['winner'].sum()
            print(f"Found {winner_count} winners ({100*winner_count/len(winners_data):.2f}% of data)")
        
        if 'tournament_id' in winners_data.columns:
            tournament_count = winners_data['tournament_id'].nunique()
            print(f"Dataset covers {tournament_count} unique tournaments")
        
        print("\nDataset sample:")
        print(winners_data.head())


if __name__ == "__main__":
    main()