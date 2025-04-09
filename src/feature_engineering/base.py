from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    
    def __init__(self, data_extractor):
        self.data_extractor = data_extractor
    
    @abstractmethod
    def extract_features(self, *args, **kwargs):
        pass