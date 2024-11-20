# app/models/naive_bayes.py

from typing import Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer, RobustScaler
from .base_model import BaseGenreModel
import logging
logger = logging.getLogger(__name__)

class GaussianNBModel(BaseGenreModel):
    """
    Gaussian Naive Bayes implementation for genre classification.
    Includes specific preprocessing optimized for Naive Bayes assumptions:
    - Power transformation to make features more Gaussian-like
    - Robust scaling to handle outliers
    - Optional feature selection based on variance
    """
    
    def __init__(self, var_smoothing: float = 1e-9, min_variance_percentile: float = 1):
        """
        Initialize Gaussian Naive Bayes model.
        
        Args:
            var_smoothing: Portion of the largest variance of all features that is 
                         added to variances for calculation stability
            min_variance_percentile: Remove features with variance below this percentile
        """
        super().__init__()
        self.model = GaussianNB(var_smoothing=var_smoothing)
        self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        self.scaler = RobustScaler(quantile_range=(5, 95))
        self.min_variance_percentile = min_variance_percentile
        self.feature_mask: Optional[np.ndarray] = None
        self.feature_variances: Optional[np.ndarray] = None
        
    def preprocess_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Preprocess data specifically for Gaussian Naive Bayes:
        - Apply power transformation to make features more Gaussian-like.
        - Scale features using robust scaling.
        - Optionally select features based on variance during training.

        Args:
            X: Input feature matrix.
            y: Target labels (only during training, for feature selection).

        Returns:
            Preprocessed feature matrix.
        """
        logger.debug(f"Preprocessing input: X={X.shape}")
        if y is not None:
            logger.debug(f"Preprocessing target: {len(y)}")
            assert len(X) == len(y), f"Mismatch in X and y: {X.shape[0]} vs {len(y)}"

        # During training
        if y is not None:
            # Apply power transform to make features more Gaussian-like
            X_transformed = self.power_transformer.fit_transform(X)
            logger.debug(f"After power transform: {X_transformed.shape}")

            # Scale features
            X_scaled = self.scaler.fit_transform(X_transformed)
            logger.debug(f"After scaling: {X_scaled.shape}")

            # Calculate feature variances
            self.feature_variances = np.var(X_scaled, axis=0)
            logger.debug(f"Feature variances calculated: {self.feature_variances}")

            # Select features based on variance
            if self.min_variance_percentile > 0:
                variance_threshold = np.percentile(self.feature_variances, self.min_variance_percentile)
                self.feature_mask = self.feature_variances >= variance_threshold
                X_selected = X_scaled[:, self.feature_mask]
                logger.debug(f"Features selected: {X_selected.shape}")
            else:
                self.feature_mask = np.ones(X_scaled.shape[1], dtype=bool)
                X_selected = X_scaled

            return X_selected

        # During inference (no target `y` provided)
        else:
            # Apply saved transformations
            X_transformed = self.power_transformer.transform(X)
            X_scaled = self.scaler.transform(X_transformed)

            # Apply feature selection if used during training
            if self.feature_mask is not None:
                X_selected = X_scaled[:, self.feature_mask]
                logger.debug(f"Features selected for inference: {X_selected.shape}")
            else:
                X_selected = X_scaled

            return X_selected


            
    def get_model_params(self) -> Dict[str, Tuple[str, Any]]:
        """
        Define hyperparameter search space for Gaussian Naive Bayes.
        
        Returns:
            Dictionary of parameter names and their valid ranges/options
        """
        return {
            'var_smoothing': ('float', (1e-11, 1e-7)),
            'min_variance_percentile': ('float', (0, 5))
        }
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Calculate feature importance based on variance.
        Higher variance indicates more discriminative power in Naive Bayes.
        
        Returns:
            Dictionary mapping feature names to their importance scores
        """
        if self.feature_variances is None or not self.is_fitted:
            return None
            
        # Get original feature names
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_variances))]
        else:
            feature_names = self.feature_names
            
        # Create importance dictionary only for selected features
        if self.feature_mask is not None:
            selected_features = [
                name for name, selected in zip(feature_names, self.feature_mask)
                if selected
            ]
            selected_variances = self.feature_variances[self.feature_mask]
        else:
            selected_features = feature_names
            selected_variances = self.feature_variances
            
        # Normalize variances to get importance scores
        importance_scores = selected_variances / np.sum(selected_variances)
        
        return dict(sorted(
            zip(selected_features, importance_scores),
            key=lambda x: x[1],
            reverse=True
        ))
        
    def get_feature_density_plot_data(self, X: np.ndarray, 
                                    y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get data for plotting feature density by class.
        Useful for visualizing the Gaussian assumption.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Dictionary with data for density plotting
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting density plot data")
            
        # Preprocess the data
        X_processed = self.preprocess_data(X)
        
        # Get unique classes
        classes = np.unique(y)
        
        # Calculate mean and std for each feature per class
        density_data = {}
        for class_label in classes:
            class_mask = y == class_label
            X_class = X_processed[class_mask]
            
            density_data[class_label] = {
                'means': np.mean(X_class, axis=0),
                'stds': np.std(X_class, axis=0)
            }
            
        return density_data