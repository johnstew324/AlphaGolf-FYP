import os
import pandas as pd
import numpy as np
import sys
# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from sklearn.ensemble import GradientBoostingRegressor
from feature_engineering.feature_selection.feature_transformer import FeatureTransformer
from feature_engineering.feature_selection.feature_analyser import FeatureAnalyzer
from feature_engineering.feature_selection.feature_selector import FeatureSelector
from feature_engineering.feature_selection.feature_refiner import FeatureRefiner


def main():
    print("===== Enhancing Winner-Specific Feature Set =====")
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    winner_file = os.path.join(current_dir, 'feature_analysis', 'winner_specific_features.csv')
    original_file = os.path.join(current_dir, 'feature_analysis', 'predictive_features.csv')
    output_dir = os.path.join(current_dir, 'feature_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load both feature sets
    print("\nLoading winner-specific features...")
    winner_df = pd.read_csv(winner_file)
    print(f"Loaded {len(winner_df)} rows with {len(winner_df.columns)} winner-specific features")
    
    print("\nLoading original predictive features...")
    original_df = pd.read_csv(original_file)
    print(f"Loaded {len(original_df)} rows with {len(original_df.columns)} original features")
    
    # Step 2: Identify problematic features to remove
    problematic_features = [
        'fedex_fall_rank',
        'has_performance_data_scaled',
        'best_finish_position',
        'off_tee_top50_count',
        'strokes_gained_off_the_tee_par_4_rank',
        'career_success_rating_numeric',
        'events_1_scaled',
        'putting_strengths',
        'first_year',
        'top_quartile_count',
        "standings_rank_1_scaled",
        "has_weather_data",
        "seconds",
        "last_year",
        "finish_position_std",
        "career_wins_1_scaled",
        "international_wins",
        "putting_bottom50_count",
        "approach_weaknesses",
        "par3_par_pct",
        "success_ratio",
        "data_completeness",
        "year",
        "others",
    ]
    
    # Add features with high missing value percentage
    missing_pct = winner_df.isnull().mean() * 100
    high_missing_features = missing_pct[missing_pct > 20].index.tolist()
    print(f"Features with >20% missing values: {len(high_missing_features)}")
    for feat in high_missing_features[:10]:  # Show top 10 for reference
        print(f"  - {feat}: {missing_pct[feat]:.2f}% missing")
    
    # Combine all features to remove
    features_to_remove = list(set(problematic_features + high_missing_features))
    
    # Keep IDs
    id_columns = ['player_id', 'tournament_id']
    features_to_remove = [f for f in features_to_remove if f not in id_columns]
    
    print(f"\nRemoving {len(features_to_remove)} problematic features")
    
    # Step 3: Find potentially strong replacements from original dataset
    # Identify all columns in original_df not in winner_df
    potential_replacements = [col for col in original_df.columns 
                              if col not in winner_df.columns and col not in features_to_remove]
    
    print(f"\nFound {len(potential_replacements)} potential replacement features")
    
    # Create target variable (if applicable)
    # For this example, we'll use a synthetic target based on career wins as a proxy
    if 'career_wins' in original_df.columns:
        original_df['winner'] = (original_df['career_wins'] > 0).astype(int)
    else:
        # Fallback to simpler approach
        winner_related = [col for col in original_df.columns if 'win' in col.lower()]
        if winner_related:
            original_df['winner'] = (original_df[winner_related[0]] > 0).astype(int)
        else:
            # Last resort random assignment
            np.random.seed(42)
            original_df['winner'] = (np.random.random(len(original_df)) < 0.05).astype(int)
    
    # Calculate importance scores for potential replacements
    print("\nCalculating importance scores for replacement candidates...")
    
    # Prepare data for importance calculation
    feature_candidates = [col for col in potential_replacements 
                          if col in original_df.columns 
                          and is_numeric_column(original_df[col])]
    
    X = original_df[feature_candidates].copy()
    # Handle missing values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].mean() if not X[col].isna().all() else 0)
    
    y = original_df['winner']
    
    # Train model for feature importance
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    try:
        model.fit(X, y)
        
        # Get feature importances
        replacement_importance = pd.DataFrame({
            'feature': feature_candidates,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top replacements
        num_to_replace = min(len(features_to_remove) + 5, len(replacement_importance))  # Add a few extra
        top_replacements = replacement_importance.head(num_to_replace)['feature'].tolist()
        
        print(f"\nSelected {len(top_replacements)} high-importance replacement features")
        print("\nTop 10 replacement features by importance:")
        for i, (_, row) in enumerate(replacement_importance.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.6f}")
    except Exception as e:
        print(f"Error calculating importances: {str(e)}")
        # Fallback to selecting replacements based on name patterns
        win_related = [col for col in feature_candidates if any(term in col.lower() 
                      for term in ['win', 'victory', 'success', 'top', 'performance'])]
        skill_related = [col for col in feature_candidates if any(term in col.lower() 
                        for term in ['strokes_gained', 'putting', 'driving', 'approach'])]
        
        # Combine and take top N
        top_replacements = (win_related + skill_related)[:len(features_to_remove) + 5]
        print(f"\nSelected {len(top_replacements)} replacement features based on naming patterns")
    
    # Step 4: Create enhanced feature set
    # Remove problematic features
    enhanced_features = [col for col in winner_df.columns if col not in features_to_remove]
    
    # Add replacement features
    for col in top_replacements:
        if col not in enhanced_features and col in original_df.columns:
            original_df[col].name = col  # Ensure column name is preserved
            winner_df[col] = original_df[col]
            enhanced_features.append(col)
    
    print(f"\nFinal enhanced feature set: {len(enhanced_features)} features")
    
    # Create enhanced dataframe
    enhanced_df = winner_df[enhanced_features]
    
    # Step 5: Save enhanced feature set
    output_file = os.path.join(output_dir, 'enhanced_winner_features.csv')
    enhanced_df.to_csv(output_file, index=False)
    
    print(f"\nSaved enhanced feature set to: {output_file}")
    
    # Print some stats about the enhanced feature set
    print("\nEnhanced feature set statistics:")
    print(f"  - Original winner features: {len(winner_df.columns)}")
    print(f"  - Features removed: {len(features_to_remove)}")
    print(f"  - Features added: {len(top_replacements)}")
    print(f"  - Final feature count: {len(enhanced_features)}")
    
    # List top new features
    if top_replacements:
        print("\nTop added features:")
        for i, feat in enumerate(top_replacements[:10], 1):  # Show top 10
            if feat in enhanced_features:
                print(f"  {i}. {feat}")
    
    print("\n===== Enhanced Winner Feature Set Creation Complete =====")

def is_numeric_column(series):
    """Check if a pandas series contains numeric data."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    
    # Check if string column can be converted to numeric
    if series.dtype == 'object' or series.dtype == 'string':
        try:
            pd.to_numeric(series)
            return True
        except:
            return False
    
    return False

if __name__ == "__main__":
    main()