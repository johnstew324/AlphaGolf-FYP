import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from feature_engineering.feature_selection.feature_transformer import FeatureTransformer
from feature_engineering.feature_selection.feature_analyser import FeatureAnalyzer
from feature_engineering.feature_selection.feature_selector import FeatureSelector

def plot_correlation_matrix(df, title, n_features=30, figsize=(15, 12), output_file=None):

    if len(df.columns) > n_features:
        variances = df.var().sort_values(ascending=False)
        top_features = variances.index[:n_features].tolist()
        df_subset = df[top_features]
    else:
        df_subset = df
        

    corr = df_subset.corr()
    
    # Create figure
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        annot=False,
        square=True,
        linewidths=.5,
        cbar_kws={'shrink': .5}
    )
    plt.title(title)
    plt.tight_layout()
    
    # Save if path provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_feature_importance(feature_names, importance_values, direction=None, title='Feature Importance', top_n=20, figsize=(12, 10), output_file=None):
    try:
        valid_indices = []
        for i in range(len(importance_values)):
            if i < len(feature_names) and not pd.isna(importance_values[i]):
                valid_indices.append(i)
        
        if not valid_indices:
            print(f"Warning: No valid importance values found for {title}")
            return None
        
        valid_features = [feature_names[i] for i in valid_indices]
        valid_importance = [importance_values[i] for i in valid_indices]
        valid_direction = [1] * len(valid_indices)  # Default to positive
        
        if direction is not None:
            valid_direction = [direction[i] if i < len(direction) else 1 for i in valid_indices]

        importance_df = pd.DataFrame({
            'feature': valid_features,
            'importance': valid_importance,
            'direction': valid_direction
        })

        importance_df['abs_importance'] = importance_df['importance'].abs()
        importance_df = importance_df.sort_values('abs_importance', ascending=False).head(top_n)

        plt.figure(figsize=figsize)
        bars = plt.barh(
            y=importance_df['feature'],
            width=importance_df['importance'],
            color=importance_df['direction'].apply(lambda x: 'green' if x > 0 else 'red')
        )
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(title)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save if path provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    except Exception as e:
        print(f"Warning: Could not create feature importance plot: {str(e)}")
        return None

def plot_feature_cluster_map(df, title="Feature Cluster Map", figsize=(15, 15), output_file=None):
    corr = df.corr()

    distance_matrix = 1 - abs(corr)
    distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
    
    try:
        linkage = hierarchy.linkage(squareform(distance_matrix), method='average')

        plt.figure(figsize=figsize)
        sns.clustermap(
            corr,
            row_linkage=linkage,
            col_linkage=linkage,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            figsize=figsize
        )
    except Exception as e:
        print(f"Warning: Could not create hierarchical cluster map: {str(e)}")
        print("Falling back to standard correlation heatmap")
        
        plt.figure(figsize=figsize)
        sns.clustermap(
            corr,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            figsize=figsize
        )

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_pca_explained_variance(df, n_components=10, figsize=(10, 6), output_file=None):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca = PCA(n_components=min(n_components, df.shape[1], df.shape[0]))
    pca.fit(scaled_data)
    plt.figure(figsize=figsize)

    plt.bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        alpha=0.8,
        color='skyblue'
    )
    plt.step(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        where='mid',
        color='red'
    )
    
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save if path provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_feature_distributions(df, top_n=10, figsize=(15, 10), output_file=None):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    variances = numeric_df.var().sort_values(ascending=False)
    top_features = variances.index[:top_n].tolist()

    plt.figure(figsize=figsize)

    nrows = (top_n + 1) // 2
    ncols = 2
    
    for i, feature in enumerate(top_features):
        plt.subplot(nrows, ncols, i+1)
        sns.histplot(numeric_df[feature].dropna(), kde=True)
        plt.title(feature)
        plt.tight_layout()
    
    # Save if path provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def analyze_multicollinearity(df, target_col=None, threshold=0.7, output_file=None):
    corr = df.corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    collinear_features = []
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        
        for other_col in high_corr:
            if target_col and (col == target_col or other_col == target_col):
                continue
                
            corr_value = upper.loc[other_col, col]
            collinear_features.append({
                'feature1': col,
                'feature2': other_col,
                'correlation': corr_value
            })
    
    result = pd.DataFrame(collinear_features).sort_values('correlation', ascending=False)

    if output_file and not result.empty:
        with open(output_file, 'w') as f:
            f.write("# Multicollinearity Analysis\n\n")
            f.write(f"Features with absolute correlation > {threshold}:\n\n")
            for _, row in result.iterrows():
                f.write(f"- {row['feature1']} <-> {row['feature2']}: {row['correlation']:.4f}\n")
    
    return result

def model_based_importance(df, target_col, method='rf', n_estimators=100, output_file=None):
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        features = [col for col in numeric_cols if col != target_col]
        
        if not features:
            print(f"Warning: No suitable numeric features found for model-based importance")
            return None

        X = df[features].fillna(df[features].mean())
        y = df[target_col]

        if method == 'rf':
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(X, y)
            importance = model.feature_importances_
        else:
            raise ValueError(f"Unknown method: {method}")

        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)

        max_importance = importance_df['importance'].max()
        if max_importance > 0:
            importance_df['importance_norm'] = importance_df['importance'] / max_importance
        else:
            importance_df['importance_norm'] = 0
        
        importance_df['direction'] = [
            1 if not pd.isna(df[feature].corr(df[target_col])) and df[feature].corr(df[target_col]) > 0 else -1
            for feature in importance_df['feature']
        ]

        if output_file:
            importance_df.to_csv(output_file, index=False)
        
        return importance_df
    
    except Exception as e:
        print(f"Warning: Could not calculate model-based importance: {str(e)}")
        return None

def main():
    print("==== Testing Improved Feature Engineering Pipeline ====")
    
    print("\nLoading test features...")
    features_df = pd.read_csv(os.path.join(current_dir, 'output', 'all_tournaments_features_20250406_190259.csv'))
    print(f"Loaded {len(features_df)} rows with {len(features_df.columns)} features")
    

    output_dir = os.path.join(current_dir, 'feature_analysis')
    os.makedirs(output_dir, exist_ok=True)
    

    print("\nTransforming features...")
    transformer = FeatureTransformer(drop_timestamps=True, special_player_ids=["35891"])
    transformed_df = transformer.fit_transform(features_df)
    print(f"Transformed data has {transformed_df.shape[1]} features")
    
    print("\nAnalyzing transformed features...")
    analyzer = FeatureAnalyzer(transformed_df)
    analysis_results = analyzer.analyze_features()
    
    
    analyzer.generate_analysis_report(os.path.join(output_dir, 'feature_analysis_report.md'))
    
    print("\nGenerating correlation matrix visualisation...")
    plot_correlation_matrix(
        transformed_df.select_dtypes(include=['int64', 'float64']),
        "Feature Correlation Matrix (Transformed Data)",
        n_features=50,
        output_file=os.path.join(output_dir, 'correlation_matrix.png')
    )
    
    print("\nApplying improved feature selection...")
    selector = FeatureSelector(transformed_df, analyzer)

    selected_df = selector.select_features(method='combined', params={
        'n_features': 50,
        'variance_threshold': 0.005,
        'corr_threshold': 0.9,
        'target_col': 'victory_potential', 
        'include_special_columns': True
    })

    print("\nFiltering out non-predictive features...")
    filtered_df = selector.filter_non_predictive_features(selected_df)
    print(f"After filtering: {filtered_df.shape[1]} features remaining")

    filtered_df.to_csv(os.path.join(output_dir, 'predictive_features.csv'), index=False)
    print(f"\nPredictive features saved to {os.path.join(output_dir, 'predictive_features.csv')}")

    print("\nSelected predictive features:")
    for feature in filtered_df.columns[:20]:
        print(f"- {feature}")

    print("\nGenerating correlation matrix for selected features...")
    plot_correlation_matrix(
        filtered_df.select_dtypes(include=['int64', 'float64']),
        "Correlation Matrix (Selected Features)",
        output_file=os.path.join(output_dir, 'selected_correlation_matrix.png')
    )

    print("\nAnalyzing multicollinearity...")
    multicollinearity = analyze_multicollinearity(
        filtered_df.select_dtypes(include=['int64', 'float64']), 
        target_col='victory_potential',
        threshold=0.8,
        output_file=os.path.join(output_dir, 'multicollinearity_report.md')
    )

    print("\nGenerating feature cluster map...")
    try:
        plot_feature_cluster_map(
            filtered_df.select_dtypes(include=['int64', 'float64']),
            "Feature Hierarchical Clustering",
            output_file=os.path.join(output_dir, 'feature_cluster_map.png')
        )
    except Exception as e:
        print(f"Warning: Feature cluster map generation failed: {str(e)}")
        print("Continuing with analysis...")
    

    print("\nGenerating feature distributions...")
    plot_feature_distributions(
        filtered_df,
        top_n=12,
        output_file=os.path.join(output_dir, 'feature_distributions.png')
    )

    if 'victory_potential' in filtered_df.columns:
        print("\nEvaluating feature importance...")
        
        try:
            numeric_features = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_features = [col for col in numeric_features if col not in ['player_id', 'tournament_id']]
            
            importances = []
            directions = []
            valid_features = []
            
            for feature in numeric_features:
                if feature != 'victory_potential' and not filtered_df[feature].isna().all():
                    try:
                        corr = filtered_df[feature].corr(filtered_df['victory_potential'])
                        if not pd.isna(corr):
                            valid_features.append(feature)
                            importances.append(abs(corr))
                            directions.append(1 if corr > 0 else -1)
                    except Exception as e:
                        print(f"Warning: Could not calculate correlation for {feature}: {str(e)}")
            
      
            print("\nGenerating correlation-based importance plot...")
            plot_feature_importance(
                valid_features,
                importances,
                directions,
                title='Feature Importance (Correlation with victory_potential)',
                output_file=os.path.join(output_dir, 'correlation_importance.png')
            )
            
   
            print("\nCalculating model-based importance...")
            model_importance = model_based_importance(
                filtered_df,
                'victory_potential',
                output_file=os.path.join(output_dir, 'model_importance.csv')
            )
            

            if model_importance is not None:
                print("\nGenerating model-based importance plot...")
                plot_feature_importance(
                    model_importance['feature'].tolist(),
                    (model_importance['importance_norm'] * model_importance['direction']).tolist(),
                    None,
                    title='Feature Importance (Random Forest)',
                    output_file=os.path.join(output_dir, 'model_importance.png')
                )
                
                print("\nTop 20 features by model-based importance:")
                for _, row in model_importance.head(20).iterrows():
                    direction = "positive" if row['direction'] > 0 else "negative"
                    print(f"- {row['feature']}: {row['importance_norm']:.4f} ({direction})")
            else:
                print("Model-based importance calculation failed, skipping related visualizations.")
        
        except Exception as e:
            print(f"Warning: Feature importance evaluation failed: {str(e)}")
    

    print("\nAnalyzing potential dimensionality reduction with PCA...")
    try:
        plot_pca_explained_variance(
            filtered_df.select_dtypes(include=['int64', 'float64']).drop(columns=['victory_potential'], errors='ignore'),
            output_file=os.path.join(output_dir, 'pca_explained_variance.png')
        )
    except Exception as e:
        print(f"Warning: PCA analysis failed: {str(e)}")
    
    print("\n==== Testing Complete ====")
    print(f"All analysis outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()