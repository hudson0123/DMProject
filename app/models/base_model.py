# app/models/base_model.py

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, cohen_kappa_score, confusion_matrix
)

class BaseGenreModel(ABC, BaseEstimator, ClassifierMixin):
    """
    Abstract base class for music genre classification models.
    Implements common functionality and defines interface for specific models.
    """
    
    def __init__(self):
        """Initialize base model components."""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        
    @abstractmethod
    def preprocess_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Model-specific preprocessing steps. Must be implemented by each model.
        
        Args:
            X: Input features
            y: Target labels (optional, only provided during training)
            
        Returns:
            Preprocessed features
        """
        pass
        
    @abstractmethod
    def get_model_params(self) -> Dict[str, Tuple[str, Any]]:
        """
        Returns model-specific parameters for hyperparameter tuning.
        Must be implemented by each model.
        
        Returns:
            Dictionary of parameter names and their valid ranges/options
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseGenreModel':
        """
        Fits the model to the training data.
        
        Args:
            X: Training features
            y: Target labels
            
        Returns:
            self: The fitted model instance
        """
        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
            
        # Preprocess the features
        X_processed = self.preprocess_data(X, y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit the model
        self.model.fit(X_processed, y_encoded)
        self.is_fitted = True
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_processed = self.preprocess_data(X)
        y_pred = self.model.predict(X_processed)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for new data.
        
        Args:
            X: Features to predict probabilities for
            
        Returns:
            Predicted class probabilities
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support probability predictions"
            )
            
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting probabilities")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_processed = self.preprocess_data(X)
        return self.model.predict_proba(X_processed)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates model performance on test data.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary containing various performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            "model": self.__class__.__name__,
            "accuracy": accuracy_score(y, y_pred),
            "precision_weighted": precision_score(y, y_pred, average='weighted'),
            "recall_weighted": recall_score(y, y_pred, average='weighted'),
            "f1_weighted": f1_score(y, y_pred, average='weighted'),
            "cohen_kappa": cohen_kappa_score(y, y_pred),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "classification_report": classification_report(
                y, y_pred, 
                output_dict=True
            )
        }
        
        # Add probability metrics if available
        if hasattr(self.model, 'predict_proba'):
            try:
                y_proba = self.predict_proba(X)
                metrics["predictions_confidence"] = {
                    "mean": np.mean(np.max(y_proba, axis=1)),
                    "std": np.std(np.max(y_proba, axis=1))
                }
            except Exception as e:
                print(f"Warning: Could not compute probability metrics: {str(e)}")
        
        return metrics

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Returns feature importance if the model supports it.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if feature importance is not supported
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
            
        importances = self.model.feature_importances_
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names
            
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))

    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"