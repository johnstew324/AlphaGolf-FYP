# modeling/pipeline.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from .model_factory import ModelFactory
from .cross_validator import ChronologicalCrossValidator
from .metrics import calculate_metrics, check_calibration

class ModelPipeline:
    """Pipeline for training and evaluating golf prediction models."""
    
    def __init__(self, model_type='gb', feature_sets=None, targets=None, params=None, output_dir='experiments'):
        """
        Initialize the pipeline.
        
        Args:
            model_type (str): Type of model to use
            feature_sets (dict): Dictionary mapping target to feature list
            targets (list): List of targets to predict
            params (dict): Model parameters
            output_dir (str): Directory to save outputs
        """
        self.model_type = model_type
        self.feature_sets = feature_sets
        self.targets = targets or ['win', 'cut', 'top3', 'top10']
        self.params = params
        self.output_dir = output_dir
        self.models = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_optimized_feature_sets(self, path):
        """
        Load optimized feature sets from JSON.
        
        Args:
            path (str): Path to JSON file with optimized feature sets
            
        Returns:
            self
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.feature_sets = data['optimized_sets']
        return self
    
    def train_model(self, target, X_train, y_train):
        """
        Train a model for a specific target.
        
        Args:
            target (str): Target to predict
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            BaseModel: Trained model
        """
        # Get feature set for this target
        feature_set = self.feature_sets.get(target) if self.feature_sets else None
        
        # Create and train model
        model = ModelFactory.create_model(self.model_type, target, feature_set, self.params)
        model.fit(X_train, y_train)
        
        # Store model
        self.models[target] = model
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred_proba)
        
        # Check calibration
        prob_true, prob_pred, calibration_error = check_calibration(y_test, y_pred_proba)
        metrics['calibration_error'] = calibration_error
        
        return metrics
    
    def run_training(self, data, target_col_prefix='', target_mapping=None):
        """
        Run full training and evaluation for all targets.
        
        Args:
            data (pd.DataFrame): Data with features and targets
            target_col_prefix (str): Prefix for target columns
            target_mapping (dict): Mapping from target name to column name
            
        Returns:
            dict: Results for all targets
        """
        results = {}
        timestamps = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for target in self.targets:
            print(f"\nTraining model for target: {target}")
            
            # Get target column, use mapping if provided
            if target_mapping and target in target_mapping:
                target_col = f"{target_col_prefix}{target_mapping[target]}"
            else:
                target_col = f"{target_col_prefix}{target}"
                
            if target_col not in data.columns:
                print(f"Warning: Target column {target_col} not found. Skipping.")
                continue
            
            # Extract features and target
            y = data[target_col]
            
            # Check if we have any positive examples
            if y.sum() == 0:
                print(f"Warning: No positive examples for {target}. Skipping.")
                continue
                
            # Drop all target-related columns
            target_cols = []
            if target_mapping:
                target_cols = [f"{target_col_prefix}{col}" for col in target_mapping.values()]
            else:
                target_cols = [col for col in data.columns if col.startswith(target_col_prefix) and any(
                    target_name in col.lower() for target_name in ['win', 'top', 'cut', 'position'])]
            
            X = data.drop(columns=target_cols, errors='ignore')
            
            # Train model on all data
            model = self.train_model(target, X, y)
            
            # Save model
            model_path = os.path.join(self.output_dir, f"{self.model_type}_{target}_{timestamps}.joblib")
            model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Save feature importance
            importance = model.get_feature_importance()
            importance_path = os.path.join(self.output_dir, f"importance_{target}_{timestamps}.csv")
            importance.to_csv(importance_path, index=False)
            
            # Store results
            results[target] = {
                'model_path': model_path,
                'importance_path': importance_path
            }
        
        # Save results overview
        results_path = os.path.join(self.output_dir, f"training_results_{timestamps}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_cross_validation(self, data, n_splits=5, target_col_prefix='', tournament_id_col='tournament_id', target_mapping=None):
        """
        Run cross-validation for all targets.
        
        Args:
            data (pd.DataFrame): Data with features and targets
            n_splits (int): Number of CV splits
            target_col_prefix (str): Prefix for target columns
            tournament_id_col (str): Column with tournament IDs
            target_mapping (dict): Mapping from target name to column name
            
        Returns:
            dict: Cross-validation results
        """
        cv_results = {}
        timestamps = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize cross-validator
        cv = ChronologicalCrossValidator(n_splits=n_splits)
        
        for target in self.targets:
            print(f"\nCross-validation for target: {target}")
            
            # Get target column, use mapping if provided
            if target_mapping and target in target_mapping:
                target_col = f"{target_col_prefix}{target_mapping[target]}"
            else:
                target_col = f"{target_col_prefix}{target}"
                
            if target_col not in data.columns:
                print(f"Warning: Target column {target_col} not found. Skipping.")
                continue
            
            # Extract features and target
            y = data[target_col]
            
            # Check if we have any positive examples
            if y.sum() == 0:
                print(f"Warning: No positive examples for {target}. Skipping.")
                continue
                
            # Drop all target-related columns
            target_cols = []
            if target_mapping:
                target_cols = [f"{target_col_prefix}{col}" for col in target_mapping.values()]
            else:
                target_cols = [col for col in data.columns if col.startswith(target_col_prefix) and any(
                    target_name in col.lower() for target_name in ['win', 'top', 'cut', 'position'])]
            
            X = data.drop(columns=target_cols, errors='ignore')
            
            # Store metrics for each fold
            fold_metrics = []
            
            # Run cross-validation by tournaments
            for fold, (train_idx, test_idx) in enumerate(cv.split_by_tournaments(data, tournament_id_col)):
                print(f"  Fold {fold+1}/{n_splits}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Check if fold has enough data
                if len(y_train) == 0 or len(y_test) == 0:
                    print(f"    Warning: Empty train or test set in fold {fold+1}. Skipping.")
                    continue
                    
                # Check if we have positive examples in both train and test
                if y_train.sum() == 0 or y_test.sum() == 0:
                    print(f"    Warning: No positive examples in train or test set in fold {fold+1}. Skipping.")
                    continue
                
                # Train model
                model = self.train_model(f"{target}_fold{fold}", X_train, y_train)
                
                # Evaluate model
                metrics = self.evaluate_model(model, X_test, y_test)
                metrics['fold'] = fold
                fold_metrics.append(metrics)
                
                print(f"    AUC: {metrics['auc']:.4f}, Log Loss: {metrics['log_loss']:.4f}")
            
            # Check if we have any fold metrics
            if not fold_metrics:
                print(f"  No valid folds for {target}. Skipping target.")
                continue
            
            # Aggregate metrics across folds
            agg_metrics = {
                'mean_auc': np.mean([m['auc'] for m in fold_metrics]),
                'std_auc': np.std([m['auc'] for m in fold_metrics]),
                'mean_log_loss': np.mean([m['log_loss'] for m in fold_metrics]),
                'std_log_loss': np.std([m['log_loss'] for m in fold_metrics]),
                'folds': fold_metrics
            }
            
            # Store results
            cv_results[target] = agg_metrics
            
            print(f"  Mean AUC: {agg_metrics['mean_auc']:.4f} (±{agg_metrics['std_auc']:.4f})")
            print(f"  Mean Log Loss: {agg_metrics['mean_log_loss']:.4f} (±{agg_metrics['std_log_loss']:.4f})")
        
        # Save CV results
        cv_results_path = os.path.join(self.output_dir, f"cv_results_{timestamps}.json")
        with open(cv_results_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        return cv_results
