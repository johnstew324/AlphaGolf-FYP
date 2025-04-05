# modeling/model_factory.py
from .models.gradient_boosting import GradientBoostingModel
# Later you'll import your DQN model here

class ModelFactory:
    """Factory class to create different types of models."""
    
    @staticmethod
    def create_model(model_type, target, feature_set=None, params=None):
        """
        Create a model of the specified type.
        
        Args:
            model_type (str): Type of model to create ('gb', 'dqn')
            target (str): Target variable to predict
            feature_set (list, optional): List of features to use
            params (dict, optional): Model parameters
            
        Returns:
            BaseModel: A model instance
        """
        if model_type.lower() == 'gb':
            return GradientBoostingModel(target, feature_set, params)
        elif model_type.lower() == 'dqn':
            # For later implementation
            raise NotImplementedError("DQN model not implemented yet")
        else:
            raise ValueError(f"Unknown model type: {model_type}")