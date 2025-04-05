# modeling/base.py
import os
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import joblib

class BaseModel(ABC):
    """Base abstract class for all golf prediction models."""
    
    def __init__(self, target, feature_set=None, params=None):
        """
        Initialize the base model.
        
        Args:
            target (str): Target variable to predict ('win', 'cut', 'top3', 'top10')
            feature_set (list, optional): List of features to use for prediction
            params (dict, optional): Model hyperparameters
        """
        self.target = target
        self.feature_set = feature_set
        self.params = params or {}
        self.model = None
        self.feature_importance = None
    
    @abstractmethod
    def fit(self, X, y):
        """
        Train the model on given data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series or np.array): Target variable
            
        Returns:
            self: Trained model instance
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Predict probability of target for given features.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.array: Predicted probabilities
        """
        pass
    
    def save(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            None
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        joblib.dump({
            'model': self.model,
            'target': self.target,
            'feature_set': self.feature_set,
            'params': self.params,
            'feature_importance': self.feature_importance
        }, path)
    
    @classmethod
    def load(cls, path):
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            BaseModel: Loaded model instance
        """
        data = joblib.load(path)
        instance = cls(data['target'], data['feature_set'], data['params'])
        instance.model = data['model']
        instance.feature_importance = data['feature_importance']
        return instance
    
    def get_feature_importance(self):
        """
        Get feature importance from the model.
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available for this model")
            
        return self.feature_importance