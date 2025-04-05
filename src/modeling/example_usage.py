# modeling/example_usage.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pipeline import ModelPipeline

def main():
    """Run an example training and prediction workflow."""
    # Load data
    print("Loading data...")
    data_path = os.path.join(parent_dir, "feature_engineering", "output", "all_tournaments_features_20250405_020448.csv")
    data = pd.read_csv(data_path)
    
    # Print column names to check what target columns are available
    print("\nAvailable columns:")
    print([col for col in data.columns if col in ['winner', 'top3', 'top10', 'made_cut']])
    print("\nTarget-related columns:")
    print([col for col in data.columns if any(target in col.lower() for target in ['win', 'top', 'cut'])])
    
    # Check if targets exist, otherwise we need to prepare them
    if 'winner' not in data.columns and 'position' in data.columns:
        print("\nCreating target variables from position data...")
        data['winner'] = (data['position'] == 1).astype(int)
        data['top3'] = (data['position'] <= 3).astype(int)
        data['top10'] = (data['position'] <= 10).astype(int)
        data['made_cut'] = (data['position'] < 100).astype(int)
    
    # Load optimized feature sets
    feature_sets_path = os.path.join(parent_dir, "feature_engineering", "feature_refinement", "optimized_feature_sets.json")
    
    # Map target names to column names
    target_mapping = {
        'win': 'winner',
        'cut': 'made_cut',
        'top3': 'top3',
        'top10': 'top10'
    }
    
    # Check which targets actually exist in the data
    available_targets = []
    for model_target, data_col in target_mapping.items():
        if data_col in data.columns:
            available_targets.append(model_target)
            print(f"Found target {model_target} (column: {data_col})")
    
    if not available_targets:
        print("No target columns found! Please check your data.")
        return
    
    # Set up pipeline
    print("\nSetting up pipeline...")
    pipeline = ModelPipeline(
        model_type='gb',
        targets=available_targets,
        output_dir=os.path.join(current_dir, 'experiments')
    )
    
    # Load optimized feature sets
    pipeline.load_optimized_feature_sets(feature_sets_path)
    
    # Run cross-validation
    print("\nRunning cross-validation...")
    cv_results = pipeline.run_cross_validation(
        data,
        n_splits=5,
        target_col_prefix='',  # Adjust based on your data
        tournament_id_col='tournament_id',
        target_mapping=target_mapping  # Pass the mapping to the pipeline
    )
    
    # Print summary
    print("\nCross-validation results summary:")
    for target, metrics in cv_results.items():
        print(f"{target.upper()}: AUC = {metrics['mean_auc']:.4f} (±{metrics['std_auc']:.4f}), Log Loss = {metrics['mean_log_loss']:.4f} (±{metrics['std_log_loss']:.4f})")
    
    # Train final models
    print("\nTraining final models...")
    training_results = pipeline.run_training(data, target_mapping=target_mapping)
    
    print("\nDone!")

if __name__ == "__main__":
    main()