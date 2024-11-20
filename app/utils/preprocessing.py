# app/utils/preprocessing.py

from typing import Dict, Optional, Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy import stats
import warnings

class DataPreprocessor:
    """
    Centralized preprocessing utilities for music genre classification.
    Handles common preprocessing tasks that can be used across different models.
    """
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        handle_outliers: bool = True,
        handle_missing: bool = True,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95
    ):
        """
        Initialize preprocessor with specified settings.
        
        Args:
            scaler_type: Type of scaler ('standard', 'robust', or 'power')
            handle_outliers: Whether to remove outliers
            handle_missing: Whether to impute missing values
            variance_threshold: Minimum variance for feature selection
            correlation_threshold: Threshold for removing correlated features
        """
        self.scaler_type = scaler_type
        self.handle_outliers = handle_outliers
        self.handle_missing = handle_missing
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        
        # Initialize components
        self.scaler = self._get_scaler()
        self.imputer = SimpleImputer(strategy='median') if handle_missing else None
        
        # State tracking
        self.feature_names: Optional[List[str]] = None
        self.removed_features: List[str] = []
        self.feature_statistics: Dict[str, Dict[str, float]] = {}
        
    def _get_scaler(self) -> Union[StandardScaler, RobustScaler, PowerTransformer]:
        """Get the appropriate scaler based on settings."""
        if self.scaler_type == 'robust':
            return RobustScaler(quantile_range=(25, 75))
        elif self.scaler_type == 'power':
            return PowerTransformer(method='yeo-johnson')
        else:
            return StandardScaler()
            
    def remove_outliers(self, X: np.ndarray, 
                       method: str = 'iqr') -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers using specified method.
        
        Args:
            X: Input features
            method: Outlier detection method ('iqr', 'zscore', or 'isolation_forest')
            
        Returns:
            Clean data and mask of non-outlier samples
        """
        if method == 'iqr':
            return self._remove_outliers_iqr(X)
        elif method == 'zscore':
            return self._remove_outliers_zscore(X)
        else:
            raise ValueError(f"Unknown outlier removal method: {method}")
            
    def _remove_outliers_iqr(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using IQR method."""
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = np.all(
            (X >= lower_bound) & (X <= upper_bound),
            axis=1
        )
        
        return X[outlier_mask], outlier_mask
        
    def _remove_outliers_zscore(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using Z-score method."""
        z_scores = stats.zscore(X, axis=0)
        outlier_mask = np.all(np.abs(z_scores) < 3, axis=1)
        return X[outlier_mask], outlier_mask
        
    def remove_low_variance_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove features with variance below threshold.
        
        Args:
            X: Input features
            
        Returns:
            Filtered features and mask of selected features
        """
        variances = np.var(X, axis=0)
        mask = variances > self.variance_threshold
        
        if self.feature_names:
            self.removed_features.extend([
                name for name, selected in zip(self.feature_names, mask)
                if not selected
            ])
            
        return X[:, mask], mask
        
    def remove_correlated_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove highly correlated features.
        
        Args:
            X: Input features
            
        Returns:
            Filtered features and mask of selected features
        """
        corr_matrix = np.corrcoef(X.T)
        mask = np.ones(X.shape[1], dtype=bool)
        
        # Find pairs of highly correlated features
        for i in range(len(corr_matrix)):
            if mask[i]:
                # Find features correlated with feature i
                correlated = np.where(np.abs(corr_matrix[i]) > self.correlation_threshold)[0]
                # Remove the features with higher mean absolute correlation
                for j in correlated:
                    if i != j and mask[j]:
                        corr_i = np.mean(np.abs(corr_matrix[i]))
                        corr_j = np.mean(np.abs(corr_matrix[j]))
                        if corr_j > corr_i:
                            mask[j] = False
                            
        if self.feature_names:
            self.removed_features.extend([
                name for name, selected in zip(self.feature_names, mask)
                if not selected
            ])
            
        return X[:, mask], mask
        
    def compute_feature_statistics(self, X: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute various statistics for each feature.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of feature statistics
        """
        stats_dict = {}
        
        for i in range(X.shape[1]):
            feature_name = self.feature_names[i] if self.feature_names else f'feature_{i}'
            stats_dict[feature_name] = {
                'mean': np.mean(X[:, i]),
                'std': np.std(X[:, i]),
                'min': np.min(X[:, i]),
                'max': np.max(X[:, i]),
                'median': np.median(X[:, i]),
                'skewness': stats.skew(X[:, i]),
                'kurtosis': stats.kurtosis(X[:, i])
            }
            
        self.feature_statistics = stats_dict
        return stats_dict
        
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                     compute_stats: bool = True) -> np.ndarray:
        """
        Fit preprocessing pipeline and transform data.
        
        Args:
            X: Input features
            compute_stats: Whether to compute feature statistics
            
        Returns:
            Transformed features
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
            
        # Handle missing values
        if self.handle_missing and self.imputer:
            X = self.imputer.fit_transform(X)
            
        # Remove outliers
        if self.handle_outliers:
            X, outlier_mask = self.remove_outliers(X)
            
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Remove low variance features
        X, variance_mask = self.remove_low_variance_features(X)
        
        # Remove correlated features
        X, correlation_mask = self.remove_correlated_features(X)
        
        # Compute statistics if requested
        if compute_stats:
            self.compute_feature_statistics(X)
            
        return X
        
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform new data using fitted preprocessing pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Handle missing values
        if self.handle_missing and self.imputer:
            X = self.imputer.transform(X)
            
        # Scale features
        X = self.scaler.transform(X)
        
        return X
        
    def get_feature_names(self) -> List[str]:
        """Get names of features after preprocessing."""
        if not self.feature_names:
            # Use the shape from the scaler since we know it's fitted
            n_features = self.scaler.n_features_in_
            return [f'feature_{i}' for i in range(n_features)]
        return [name for name in self.feature_names if name not in self.removed_features]
        
    def get_preprocessing_summary(self) -> Dict[str, Union[str, List[str], Dict[str, Dict[str, float]]]]:
        """
        Get summary of preprocessing operations.
        
        Returns:
            Dictionary containing preprocessing information:
            - scaler_type: Type of scaler used
            - removed_features: List of features removed during preprocessing
            - feature_statistics: Dictionary of feature statistics
            - remaining_features: List of features after preprocessing
        """
        return {
            'scaler_type': self.scaler_type,
            'removed_features': self.removed_features,
            'feature_statistics': self.feature_statistics,
            'remaining_features': self.get_feature_names()
        }