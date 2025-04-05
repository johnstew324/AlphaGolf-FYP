# modeling/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

def calculate_metrics(y_true, y_pred_proba):
    """
    Calculate various performance metrics for probability predictions.
    
    Args:
        y_true (np.array): True binary labels
        y_pred_proba (np.array): Predicted probabilities
        
    Returns:
        dict: Dictionary of metrics
    """
    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    logloss = log_loss(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Calculate custom golf metrics
    kelly_criterion = calculate_kelly_criterion(y_true, y_pred_proba)
    
    return {
        'auc': auc,
        'log_loss': logloss,
        'brier_score': brier,
        'kelly_criterion': kelly_criterion
    }

def calculate_kelly_criterion(y_true, y_pred_proba, implied_odds=None):
    """
    Calculate Kelly Criterion for betting applications.
    
    Args:
        y_true (np.array): True binary labels
        y_pred_proba (np.array): Predicted probabilities
        implied_odds (np.array, optional): Implied odds from the market
        
    Returns:
        float: Average Kelly stake percentage
    """
    # If implied odds not provided, assume fair odds based on predicted probabilities
    if implied_odds is None:
        implied_odds = 1.0 / y_pred_proba
    
    # Calculate Kelly stake for each prediction
    kelly_stakes = []
    for i in range(len(y_true)):
        true_prob = y_pred_proba[i]
        decimal_odds = implied_odds[i]
        
        # Kelly formula: f* = (bp - q) / b = (p - q/b) / (1 - q/b)
        # where p is probability of winning, q is probability of losing,
        # and b is the decimal odds - 1
        if decimal_odds > 1.0:
            b = decimal_odds - 1.0
            q = 1.0 - true_prob
            f_star = (b * true_prob - q) / b if b * true_prob > q else 0
            kelly_stakes.append(f_star)
    
    # Return average Kelly stake
    return np.mean(kelly_stakes) if kelly_stakes else 0.0

def check_calibration(y_true, y_pred_proba, n_bins=10):
    """
    Check calibration of probability predictions.
    
    Args:
        y_true (np.array): True binary labels
        y_pred_proba (np.array): Predicted probabilities
        n_bins (int): Number of bins for calibration curve
        
    Returns:
        tuple: (prob_true, prob_pred, calibration_error)
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    # Calculate calibration error (Mean squared error between predicted and actual probabilities)
    calibration_error = np.mean((prob_true - prob_pred) ** 2)
    
    return prob_true, prob_pred, calibration_error