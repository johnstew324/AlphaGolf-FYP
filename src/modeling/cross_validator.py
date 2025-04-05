# modeling/cross_validator.py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class ChronologicalCrossValidator:
    """Cross-validation that respects chronological order of tournaments."""
    
    def __init__(self, n_splits=5):
        """
        Initialize the cross-validator.
        
        Args:
            n_splits (int): Number of folds
        """
        self.n_splits = n_splits
    
    def split(self, data, date_column='tournament_date'):
        """
        Generate train/test splits respecting chronological order.
        
        Args:
            data (pd.DataFrame): Data to split
            date_column (str): Column containing dates
            
        Yields:
            tuple: (train_idx, test_idx)
        """
        # Sort data by date
        sorted_data = data.sort_values(date_column)
        indices = np.arange(len(sorted_data))
        
        # Calculate fold size
        fold_size = len(indices) // self.n_splits
        
        # Generate splits
        for i in range(self.n_splits):
            if i < self.n_splits - 1:
                test_start = i * fold_size
                test_end = (i + 1) * fold_size
            else:
                # Last fold might have more elements
                test_start = i * fold_size
                test_end = len(indices)
                
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
            
            yield train_idx, test_idx
    
    def split_by_tournaments(self, data, tournament_id_column='tournament_id'):
        """
        Generate train/test splits by tournament groups.
        
        Args:
            data (pd.DataFrame): Data to split
            tournament_id_column (str): Column containing tournament IDs
            
        Yields:
            tuple: (train_idx, test_idx)
        """
        # Get unique tournaments in chronological order
        tournaments = data[tournament_id_column].unique()
        
        # Create a tournament-to-index mapping
        tournament_indices = {}
        for tournament in tournaments:
            tournament_indices[tournament] = data[data[tournament_id_column] == tournament].index.tolist()
        
        # Calculate fold size
        fold_size = len(tournaments) // self.n_splits
        
        # Generate splits
        for i in range(self.n_splits):
            if i < self.n_splits - 1:
                test_tournaments = tournaments[i * fold_size:(i + 1) * fold_size]
            else:
                # Last fold might have more tournaments
                test_tournaments = tournaments[i * fold_size:]
                
            test_idx = []
            for tournament in test_tournaments:
                test_idx.extend(tournament_indices[tournament])
                
            train_idx = []
            for tournament in tournaments:
                if tournament not in test_tournaments:
                    train_idx.extend(tournament_indices[tournament])
            
            yield np.array(train_idx), np.array(test_idx)