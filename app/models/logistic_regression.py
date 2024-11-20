# app/models/logistic_regression.py

from typing import Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel, RFE
import warnings
from .base_model import BaseGenreModel

class LogisticRegressionModel(BaseGenreModel):
    """
    Logistic Regression implementation for genre classification.
    
    Features:
    - L1/L2 regularization support
    - Polynomial feature generation
    - Recursive feature elimination
    - Automatic handling of multicollinearity
    - Feature importance based on coefficients
    """
    
    def __init__(
        self,
        C: float = 1.0,
        penalty: str = 'l2',
        poly_degree: int = 2,
        n_features_to_select: Optional[int] = None,
        solver: str = 'saga',
        max_iter: int = 1000
    ):
        """
        Initialize Logistic Regression model.
        
        Args:
            C: Inverse of regularization strength
            penalty: Regularization type ('l1', 'l2', or 'elasticnet')
            poly_degree: Degree of polynomial features
            n_features_to_select: Number of features to select using RFE
            solver: Algorithm to use for optimization
            max_iter: Maximum number of iterations for solver
        """
        super().__init__()
        
        # Handle warning for convergence
        warnings.filterwarnings('ignore', category=UserWarning)
        
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            multi_class='multinomial',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.poly_degree = poly_degree
        self.n_features_to_select = n_features_to_select
        
        self.poly_transformer: Optional[PolynomialFeatures] = None
        self.feature_selector: Optional[RFE] = None
        self.selected_feature_mask: Optional[np.ndarray] = None
        self.feature_names_poly: Optional[list] = None
        
    def _generate_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate polynomial features up to specified degree.
        
        Args:
            X: Input features
            
        Returns:
            Array with polynomial features
        """
        if self.poly_transformer is None:
            self.poly_transformer = PolynomialFeatures(
                degree=self.poly_degree,
                include_bias=False,
                interaction_only=True
            )
            
        if self.is_fitted:
            return self.poly_transformer.transform(X)
        else:
            X_poly = self.poly_transformer.fit_transform(X)
            # Store polynomial feature names
            if self.feature_names is not None:
                self.feature_names_poly = (
                    self.poly_transformer.get_feature_names_out(self.feature_names)
                )
            return X_poly
            
    def _handle_multicollinearity(self, X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """
        Remove highly correlated features.
        
        Args:
            X: Input features
            threshold: Correlation threshold for removal
            
        Returns:
            Array with uncorrelated features
        """
        if X.shape[1] < 2:
            return X
            
        # Calculate correlation matrix
        corr_matrix = np.abs(np.corrcoef(X.T))
        
        # Find pairs of highly correlated features
        upper_tri = np.triu(corr_matrix, k=1)
        to_drop = set()
        
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if upper_tri[i, j] > threshold:
                    # Drop the feature with higher mean correlation
                    mean_corr_i = np.mean(np.abs(corr_matrix[i]))
                    mean_corr_j = np.mean(np.abs(corr_matrix[j]))
                    to_drop.add(j if mean_corr_j > mean_corr_i else i)
        
        # Keep track of removed features
        self.removed_features = list(to_drop)
        
        # Create mask for remaining features
        keep_mask = ~np.isin(np.arange(X.shape[1]), list(to_drop))
        
        # Update feature names if available
        if self.feature_names_poly is not None:
            self.feature_names_poly = [
                name for i, name in enumerate(self.feature_names_poly)
                if i not in to_drop
            ]
        
        return X[:, keep_mask]
        
    def preprocess_data(self, X: Union[np.ndarray, pd.DataFrame],
                       y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Preprocess data specifically for Logistic Regression:
        1. Scale features
        2. Generate polynomial features
        3. Handle multicollinearity
        4. Select features using RFE
        
        Args:
            X: Input features
            y: Target labels (only used during training)
            
        Returns:
            Preprocessed features
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # During training
        if y is not None:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Generate polynomial features
            X_poly = self._generate_polynomial_features(X_scaled)
            
            # Handle multicollinearity
            X_uncorr = self._handle_multicollinearity(X_poly)
            
            # Feature selection using RFE if specified
            if self.n_features_to_select is not None:
                self.feature_selector = RFE(
                    estimator=self.model,
                    n_features_to_select=self.n_features_to_select,
                    step=0.1
                )
                X_selected = self.feature_selector.fit_transform(X_uncorr, y)
                self.selected_feature_mask = self.feature_selector.support_
                return X_selected
                
            return X_uncorr
            
        # During prediction
        else:
            X_scaled = self.scaler.transform(X)
            X_poly = self._generate_polynomial_features(X_scaled)
            X_uncorr = self._handle_multicollinearity(X_poly)
            
            if self.feature_selector is not None:
                return self.feature_selector.transform(X_uncorr)
                
            return X_uncorr
            
    def get_model_params(self) -> Dict[str, Tuple[str, Any]]:
        """
        Define hyperparameter search space for Logistic Regression.
        
        Returns:
            Dictionary of parameter names and their valid ranges/options
        """
        return {
            'C': ('float', (0.001, 10.0)),
            'penalty': ('categorical', ['l1', 'l2', 'elasticnet']),
            'l1_ratio': ('float', (0.0, 1.0)),
            'poly_degree': ('int', (1, 3)),
            'n_features_to_select': ('int', (10, 50))
        }
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance based on model coefficients.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or not hasattr(self.model, 'coef_'):
            return None
            
        # Get absolute mean coefficients across all classes
        importances = np.abs(self.model.coef_).mean(axis=0)
        
        # Get feature names
        if self.feature_names_poly is not None:
            feature_names = self.feature_names_poly
        else:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            
        # If feature selection was used, only include selected features
        if self.selected_feature_mask is not None:
            feature_names = [
                name for name, selected in zip(feature_names, self.selected_feature_mask)
                if selected
            ]
            importances = importances[self.selected_feature_mask]
            
        # Create and sort importance dictionary
        importance_dict = dict(zip(feature_names, importances))
        return dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
    def get_class_coefficients(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get coefficients for each class and feature.
        
        Returns:
            Nested dictionary mapping classes to their feature coefficients
        """
        if not self.is_fitted or not hasattr(self.model, 'coef_'):
            return None
            
        coef_dict = {}
        classes = self.label_encoder.classes_
        
        # Get feature names
        if self.feature_names_poly is not None:
            feature_names = self.feature_names_poly
        else:
            feature_names = [f'feature_{i}' for i in range(self.model.coef_.shape[1])]
            
        # If feature selection was used, only include selected features
        if self.selected_feature_mask is not None:
            feature_names = [
                name for name, selected in zip(feature_names, self.selected_feature_mask)
                if selected
            ]
            
        # Create dictionary for each class
        for idx, class_name in enumerate(classes):
            coef_dict[class_name] = dict(zip(feature_names, self.model.coef_[idx]))
            
        return coef_dict