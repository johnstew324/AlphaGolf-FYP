# modeling/data_explorer.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def main():
    """Explore the dataset to understand target variables."""
    # Load data
    print("Loading data...")
    data_path = os.path.join(parent_dir, "feature_engineering", "output", "all_tournaments_features_20250405_020448.csv")
    data = pd.read_csv(data_path)
    
    print(f"\nDataset shape: {data.shape}")
    
    # Check for target-related columns
    target_cols = [col for col in data.columns if any(
        target_name in col.lower() for target_name in ['win', 'top', 'cut', 'position'])]
    
    print("\nPotential target columns:")
    for col in target_cols:
        print(f"  - {col}")
        
    # Check for tournament identifier columns
    tournament_cols = [col for col in data.columns if 'tournament' in col.lower()]
    print("\nTournament-related columns:")
    for col in tournament_cols:
        print(f"  - {col}")
    
    # Check for position-related columns to derive targets
    position_cols = [col for col in data.columns if 'position' in col.lower()]
    if position_cols:
        print("\nPosition-related columns:")
        for col in position_cols:
            print(f"  - {col}")
            if col in data.columns:
                print(f"    Unique values: {data[col].nunique()}")
                print(f"    Min: {data[col].min()}, Max: {data[col].max()}")
                print(f"    Sample: {data[col].head(5).tolist()}")
    
    # Check tournament distribution
    if 'tournament_id' in data.columns:
        print("\nTournament distribution:")
        tournament_counts = data['tournament_id'].value_counts()
        print(f"  Number of tournaments: {len(tournament_counts)}")
        print(f"  Avg players per tournament: {tournament_counts.mean():.1f}")
        print(f"  Example tournaments: {tournament_counts.head(5)}")
    
    # Missing values in potential target columns
    print("\nMissing values in potential target columns:")
    for col in target_cols:
        if col in data.columns:
            missing = data[col].isna().sum()
            print(f"  - {col}: {missing} missing values ({missing/len(data)*100:.1f}%)")
    
    # Create sample target variables if needed
    if 'position' in data.columns:
        print("\nSample derived target variables:")
        pos_sample = data['position'].dropna().sample(min(5, len(data)))
        for pos in pos_sample:
            print(f"  Position {pos}: win={(pos==1)}, top3={(pos<=3)}, top10={(pos<=10)}, made_cut={(pos<100)}")
    
    print("\nDone exploring data!")

if __name__ == "__main__":
    main()