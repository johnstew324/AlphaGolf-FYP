import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_examine_data(filepath):
    """
    Load the dataset and examine its properties
    """
    # Load data
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"\nNumber of tournaments: {df['tournament_id'].nunique()}")
    print(f"\nNumber of players: {df['player_id'].nunique()}")
    
    # Examine tournament_id format
    print("\nSample tournament_id format:")
    print(df['tournament_id'].iloc[:5])
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_pct
    })
    print("\nColumns with missing values:")
    print(missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False))
    
    # Check for low variance features on numerical columns only
    variance = df.select_dtypes(include=['float64', 'int64']).var()
    variance_df = pd.DataFrame({
        'Variance': variance
    })
    low_variance_cols = variance_df[variance_df['Variance'] < 0.01].index.tolist()
    print("\nLow variance columns (< 0.01):")
    print(low_variance_cols)
    
    # Identify already normalized features based on naming patterns
    normalized_pattern = ['_scaled', '_norm', '_pct', '_ratio', 'percentage']
    already_normalized = [col for col in df.columns if any(pattern in col.lower() for pattern in normalized_pattern)]
    print("\nFeatures that appear to be already normalized based on naming convention:")
    print(already_normalized)
    
    # Check if any non-scaled features remain
    # Typically, scaled features are in the range (-3, 3) for standardized or (0, 1) for min-max
    potential_unscaled = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        # Skip IDs and already normalized features
        if col in ['player_id', 'tournament_id'] or any(pattern in col.lower() for pattern in normalized_pattern):
            continue
            
        col_max = df[col].max()
        col_min = df[col].min()
        if col_max > 10 or col_min < -10:
            potential_unscaled.append(col)
    
    print("\nPotentially unscaled numerical features:")
    print(potential_unscaled)
    
    return df, missing_df, low_variance_cols, potential_unscaled

def clean_and_prepare_data(df, missing_threshold=50, variance_threshold=0.01):
    """
    Clean and prepare the data for the DQN model
    """
    print("\nCleaning and preparing data...")
    
    # Create a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Get columns to drop (high missing values, low variance)
    cols_to_drop = []
    
    # Identify columns with missing values above threshold
    missing_values = clean_df.isnull().sum()
    missing_pct = (missing_values / len(clean_df)) * 100
    high_missing_cols = missing_pct[missing_pct > missing_threshold].index.tolist()
    cols_to_drop.extend(high_missing_cols)
    
    # Identify low variance columns
    variance = clean_df.var(numeric_only=True)
    low_variance_cols = variance[variance < variance_threshold].index.tolist()
    cols_to_drop.extend([col for col in low_variance_cols if col not in cols_to_drop])
    
    print(f"Dropping {len(cols_to_drop)} columns due to high missing values or low variance:")
    print(cols_to_drop)
    
    # Drop identified columns
    if cols_to_drop:
        clean_df = clean_df.drop(columns=cols_to_drop)
    
    # For remaining columns with some missing values, impute with median
    columns_with_missing = clean_df.columns[clean_df.isnull().any()].tolist()
    for col in columns_with_missing:
        if col not in ['player_id', 'tournament_id']:
            median_value = clean_df[col].median()
            clean_df[col] = clean_df[col].fillna(median_value)
            print(f"Filled missing values in {col} with median: {median_value}")
    
    # Identify columns that need normalization
    # Exclude ID columns and already normalized features (those with '_scaled' suffix)
    numerical_cols = clean_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Specifically exclude tournament_id which may have format like 'R2025016'
    # Also exclude columns that are already scaled based on naming patterns
    exclude_patterns = ['_scaled', '_norm', '_pct', '_ratio', 'percentage']
    cols_to_scale = []
    
    for col in numerical_cols:
        # Skip ID columns and columns that are already normalized based on name
        if col in ['player_id', 'tournament_id']:
            continue
        
        # Skip columns that match normalized naming patterns
        if any(pattern in col.lower() for pattern in exclude_patterns):
            print(f"Skipping normalization for column '{col}' as it appears to be already normalized")
            continue
        
        # Check the data range to determine if scaling is needed
        col_max = clean_df[col].max()
        col_min = clean_df[col].min()
        
        if col_max > 10 or col_min < -10:
            cols_to_scale.append(col)
    
    if cols_to_scale:
        print(f"\nScaling {len(cols_to_scale)} features that appear unscaled:")
        print(cols_to_scale)
        
        scaler = StandardScaler()
        clean_df[cols_to_scale] = scaler.fit_transform(clean_df[cols_to_scale])
    
    return clean_df

def create_tournament_splits(df, holdout_pct=0.2, test_pct=0.2, random_state=42):
    """
    Split data by tournaments for training, testing and holdout
    """
    # Get unique tournament IDs
    tournament_ids = df['tournament_id'].unique()
    
    # Split tournaments for holdout set
    train_test_tournaments, holdout_tournaments = train_test_split(
        tournament_ids, 
        test_size=holdout_pct, 
        random_state=random_state
    )
    
    # Further split for train and test sets
    train_tournaments, test_tournaments = train_test_split(
        train_test_tournaments,
        test_size=test_pct,
        random_state=random_state
    )
    
    # Create dataframes for each set
    train_df = df[df['tournament_id'].isin(train_tournaments)]
    test_df = df[df['tournament_id'].isin(test_tournaments)]
    holdout_df = df[df['tournament_id'].isin(holdout_tournaments)]
    
    print(f"\nData split complete:")
    print(f"Training set: {len(train_tournaments)} tournaments, {len(train_df)} rows")
    print(f"Testing set: {len(test_tournaments)} tournaments, {len(test_df)} rows")
    print(f"Holdout set: {len(holdout_tournaments)} tournaments, {len(holdout_df)} rows")
    
    return train_df, test_df, holdout_df, train_tournaments, test_tournaments, holdout_tournaments

def identify_winners(df):
    """
    Identify the winner in each tournament or create a synthetic winner
    """
    # Check if there's a winner column or indicator
    potential_winner_cols = ['top_10', 'top10', 'wins_current', 'third']
    
    # If we don't have direct winner information, we'll use standings or best finish as a proxy
    # For demonstration, let's assume the player with the best standing in each tournament is the winner
    
    # Check if we have a standings column
    if 'standings_rank_2_scaled' in df.columns:
        print("\nUsing 'standings_rank_2_scaled' to identify tournament winners")
        
        # Group by tournament and find the player with the best (lowest) standing
        winners = df.loc[df.groupby('tournament_id')['standings_rank_2_scaled'].idxmin()]
        
        # Create a binary target column
        df['is_winner'] = 0
        df.loc[winners.index, 'is_winner'] = 1
        
        # Check if we have a reasonable number of winners
        winner_count = df['is_winner'].sum()
        tournament_count = df['tournament_id'].nunique()
        
        print(f"Identified {winner_count} winners across {tournament_count} tournaments")
        
        if winner_count < tournament_count:
            print(f"Warning: Only found winners for {winner_count}/{tournament_count} tournaments")
            print("Some tournaments may not have a clear winner in the data")
    else:
        # If we don't have standings, create a synthetic winner based on available metrics
        print("\nNo direct standings found. Creating synthetic winners using available metrics")
        
        # Use a combination of metrics to identify likely winners
        # For example: high top10, low scoring average, high strokes gained
        
        # Create a 'winner_score' as a combination of important features
        key_metrics = []
        
        if 'strokes_gained_scoring_sg_total_total_sg:t' in df.columns:
            key_metrics.append('strokes_gained_scoring_sg_total_total_sg:t')
        
        if 'scoring_scoring_average_actual' in df.columns:
            # Lower is better for scoring average, so we'll negate it
            df['neg_scoring_avg'] = -df['scoring_scoring_average_actual']
            key_metrics.append('neg_scoring_avg')
        
        if 'top10' in df.columns or 'top_10' in df.columns:
            top10_col = 'top10' if 'top10' in df.columns else 'top_10'
            key_metrics.append(top10_col)
        
        if key_metrics:
            # Normalize each metric for equal weighting
            for metric in key_metrics:
                if metric in df.columns:
                    df[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
            
            # Create a composite score
            norm_metrics = [f'{metric}_norm' for metric in key_metrics if f'{metric}_norm' in df.columns]
            df['winner_score'] = df[norm_metrics].sum(axis=1)
            
            # Identify the winner in each tournament as the player with the highest winner_score
            winners = df.loc[df.groupby('tournament_id')['winner_score'].idxmax()]
            
            # Create binary target
            df['is_winner'] = 0
            df.loc[winners.index, 'is_winner'] = 1
            
            print(f"Created synthetic winners for {len(winners)} tournaments using metrics: {key_metrics}")
        else:
            print("No suitable metrics found to identify winners. Please add a winner column to your dataset.")
            df['is_winner'] = 0
    
    return df

def create_win_percentages(df):
    """
    Calculate win percentages for each player based on historical data
    """
    # Group by player_id and calculate win percentage
    player_stats = df.groupby('player_id').agg(
        total_tournaments=('tournament_id', 'nunique'),
        wins=('is_winner', 'sum')
    )
    
    player_stats['win_percentage'] = (player_stats['wins'] / player_stats['total_tournaments']) * 100
    
    # Merge back to the original dataframe
    df = df.merge(player_stats[['win_percentage']], on='player_id', how='left')
    
    print("\nCalculated historical win percentages for each player")
    
    # Show distribution of win percentages
    print("\nWin percentage distribution:")
    print(player_stats['win_percentage'].describe())
    
    return df

def prepare_data_for_dqn(filepath):
    """
    Complete data preparation pipeline for DQN model
    """
    # Load and examine data
    df, missing_df, low_variance_cols, potential_unscaled = load_and_examine_data(filepath)
    
    # Clean and prepare data
    clean_df = clean_and_prepare_data(df)
    
    # Create dataset splits
    train_df, test_df, holdout_df, train_tournaments, test_tournaments, holdout_tournaments = create_tournament_splits(clean_df)
    
    # Identify winners and create target variable
    train_df = identify_winners(train_df)
    test_df = identify_winners(test_df)
    holdout_df = identify_winners(holdout_df)
    
    # Calculate win percentages
    train_df = create_win_percentages(train_df)
    test_df = create_win_percentages(test_df)
    holdout_df = create_win_percentages(holdout_df)
    
    print("\nData preparation complete!")
    
    return train_df, test_df, holdout_df

# Example usage
if __name__ == "__main__":
    filepath = "enhanced_winner_features.csv"
    train_df, test_df, holdout_df = prepare_data_for_dqn(filepath)
    
    # Save prepared datasets
    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)
    holdout_df.to_csv("holdout_data.csv", index=False)
    
    print("\nPrepared datasets saved to CSV files.")
    
    # Quick analysis of the target variable distribution
    print("\nTarget variable distribution in training set:")
    print(train_df['is_winner'].value_counts())
    print(f"Winner percentage: {train_df['is_winner'].mean() * 100:.2f}%")