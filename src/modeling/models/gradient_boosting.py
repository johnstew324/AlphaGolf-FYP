# modeling/models/gradient_boosting.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from ..base import BaseModel

class GradientBoostingModel(BaseModel):
    """Gradient Boosting model for golf predictions."""
    
    def __init__(self, target, feature_set=None, params=None):
        """
        Initialize the Gradient Boosting model.
        
        Args:
            target (str): Target variable to predict ('win', 'cut', 'top3', 'top10')
            feature_set (list, optional): List of features to use for prediction
            params (dict, optional): GradientBoostingClassifier parameters
        """
        super().__init__(target, feature_set, params)
        
        # Set default parameters if none provided
        if not self.params:
            self.params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 0.8,
                'random_state': 42
            }
            
        # Initialize the model
        self.model = GradientBoostingClassifier(**self.params)
    
    def fit(self, X, y):
        """
        Train the Gradient Boosting model on given data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series or np.array): Target variable (binary)
            
        Returns:
            self: Trained model instance
        """
        # Filter to use only the selected feature set
        if self.feature_set:
            valid_features = [f for f in self.feature_set if f in X.columns]
            if len(valid_features) < len(self.feature_set):
                missing = set(self.feature_set) - set(valid_features)
                print(f"Warning: {len(missing)} features not found in data: {missing}")
            X = X[valid_features]
        
        # Train the model
        self.model.fit(X, y)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probability of target for given features.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.array: Predicted probability of positive class (column 1)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Filter to use only the selected feature set
        if self.feature_set:
            valid_features = [f for f in self.feature_set if f in X.columns]
            X = X[valid_features]
        
        # Return probability of positive class
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, top_n=None):
        """
        Get feature importance from the model.
        
        Args:
            top_n (int, optional): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet")
            
        if top_n:
            return self.feature_importance.head(top_n)
        return self.feature_importance