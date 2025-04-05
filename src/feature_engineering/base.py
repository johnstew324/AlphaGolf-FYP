# feature_engineering/base.py
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """Base class for all feature processors."""
    
    def __init__(self, data_extractor):
        self.data_extractor = data_extractor
    
    @abstractmethod
    def extract_features(self, *args, **kwargs):
        """Extract features from the data source."""
        pass