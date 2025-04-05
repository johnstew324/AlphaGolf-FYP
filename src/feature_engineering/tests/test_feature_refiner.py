# test_feature_refiner.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import components
from feature_engineering.feature_selection.feature_transformer import FeatureTransformer
from feature_engineering.feature_selection.feature_analyser import FeatureAnalyzer
from feature_engineering.feature_selection.feature_selector import FeatureSelector
from feature_engineering.feature_selection.feature_refiner import EnhancedFeatureRefiner

def main():
    print("==== Testing Feature Refiner ====")
    
    # Step 1: Load the test data
    print("\nLoading test features...")
    features_df = pd.read_csv(os.path.join(current_dir, 'test_features.csv'))
    print(f"Loaded {len(features_df)} rows with {len(features_df.columns)} features")
    
    # Step 2: Transform features
    print("\nTransforming features...")
    transformer = FeatureTransformer(drop_timestamps=True)
    transformed_df = transformer.fit_transform(features_df)
    print(f"Transformed data has {transformed_df.shape[1]} features")
    
    # Step 3: Analyze transformed features
    print("\nAnalyzing transformed features...")
    analyzer = FeatureAnalyzer(transformed_df)
    analyzer.analyze_features()
    
    # Step 4: Apply feature selection
    print("\nApplying feature selection...")
    selector = FeatureSelector(transformed_df, analyzer)
    selected_df = selector.select_features(method='combined')
    
    # Step 5: Apply feature refinement
    print("\nRefining features...")
    refiner = EnhancedFeatureRefiner(analyzer, selector)
    
    # Identify core features
    print("\nIdentifying core features...")
    core_features = refiner.identify_core_features(
        importance_threshold=0.3, 
        correlation_threshold=0.8,
        max_features=50
    )
    print(f"Identified {len(core_features)} core features")
    
    # Get correlated groups from analyzer
    if 'correlated_feature_groups' in analyzer.analysis_results:
        correlated_groups = analyzer.analysis_results['correlated_feature_groups']
        
        # Select representatives from correlated groups
        print("\nSelecting group representatives...")
        representatives = refiner.select_group_representatives(
            correlated_groups, 
            target_col='victory_potential'
        )
        
        print(f"Selected representatives from {len(representatives)} feature groups")
        
        # Print a sample of representatives
        print("\nSample group representatives:")
        for group, features in list(representatives.items())[:3]:
            print(f"- {group}: {', '.join(features)}")
    
    # Create interaction features
    print("\nCreating interaction features...")
    with_interactions = refiner.create_interaction_features(
        selected_df, 
        base_features=list(core_features),
        target_col='victory_potential'
    )
    
    new_interactions = set(with_interactions.columns) - set(selected_df.columns)
    print(f"Created {len(new_interactions)} interaction features")
    
    # If target data available, create target-specific features and evaluate stability
    if 'victory_potential' in transformed_df.columns:
        # Create output directory
        output_dir = os.path.join(current_dir, 'feature_refinement')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get target-specific features
        print("\nCreating target-specific feature sets...")
        target_specific = refiner.create_target_specific_features(
            transformed_df,
            methods={'win': 'correlation'}
        )
        
        for target, features in target_specific.items():
            if features:
                print(f"- {target}: {len(features)} features")
        
        # Evaluate feature stability
        if 'victory_potential' in transformed_df.columns:
            print("\nEvaluating feature stability...")
            try:
                # Check if we have valid numeric data in the target column
                valid_target = pd.to_numeric(transformed_df['victory_potential'], errors='coerce')
                if valid_target.notna().sum() > 10:  # At least 10 valid values
                    stability = refiner.evaluate_feature_stability(
                        transformed_df,
                        'victory_potential',
                        feature_subset=list(core_features),
                        output_dir=output_dir
                    )
                    
                    if not stability.empty:
                        print("\nTop 10 features by stability:")
                        for _, row in stability.head(10).iterrows():
                            print(f"- {row['feature']}: {row['mean_importance']:.4f} (±{row['std_importance']:.4f})")
                    else:
                        print("No stability results available.")
                else:
                    print(f"Skipping stability evaluation - insufficient valid numeric values in target")
            except Exception as e:
                print(f"Error during stability evaluation: {str(e)}")
        else:
            print("\nSkipping feature stability evaluation - target variable not found")
        
        # Get optimized feature sets
        print("\nGenerating optimized feature sets...")
        for target in ['win', 'cut', 'top3', 'top10']:
            optimized = refiner.get_optimized_feature_set(
                target_type=target,
                include_interactions=True,
                max_features=50
            )
            print(f"- {target}: {len(optimized)} features")
        
        # Save feature sets
        refiner.save_feature_sets(os.path.join(output_dir, 'optimized_feature_sets.json'))
        print(f"\nSaved optimized feature sets to {os.path.join(output_dir, 'optimized_feature_sets.json')}")
    
    print("\n==== Testing Complete ====")

if __name__ == "__main__":
    main()