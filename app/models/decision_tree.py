# app/models/decision_tree.py

from typing import Dict, Optional, Tuple, Any, Union, List
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from .base_model import BaseGenreModel

class DecisionTreeModel(BaseGenreModel):
    """
    Decision Tree implementation for genre classification.
    
    Features:
    - Custom splitting criteria
    - Tree visualization generation
    - Path analysis for predictions
    - Feature importance based on information gain
    - Automatic pruning strategies
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        class_weight: str = 'balanced',
        ccp_alpha: float = 0.0
    ):
        """
        Initialize Decision Tree model.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
            criterion: Splitting criterion ('gini' or 'entropy')
            class_weight: Class weight strategy
            ccp_alpha: Complexity parameter for pruning
        """
        super().__init__()
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_threshold_map: Dict[str, List[float]] = {}
        
    def preprocess_data(self, X: Union[np.ndarray, pd.DataFrame],
                       y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Preprocess data specifically for Decision Tree:
        1. Scale features
        2. Store feature thresholds
        
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
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled
            
        # During prediction
        return self.scaler.transform(X)
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> 'DecisionTreeModel':
        """
        Fit the model and store feature thresholds.
        
        Args:
            X: Training features
            y: Target labels
            
        Returns:
            self: The fitted model
        """
        # Regular fit
        super().fit(X, y)
        
        # Store feature thresholds
        self._store_feature_thresholds()
        
        return self
        
    def _store_feature_thresholds(self):
        """Store the threshold values used for each feature in the tree."""
        if not self.is_fitted:
            return
            
        # Get feature names
        feature_names = (self.feature_names if self.feature_names is not None 
                        else [f'feature_{i}' for i in range(self.model.n_features_in_)])
        
        # Initialize threshold map
        self.feature_threshold_map = {name: [] for name in feature_names}
        
        # Get threshold values for each feature
        tree = self.model.tree_
        for feature_idx in range(len(feature_names)):
            # Find all nodes that split on this feature
            feature_nodes = np.where(tree.feature == feature_idx)[0]
            if len(feature_nodes) > 0:
                thresholds = tree.threshold[feature_nodes]
                self.feature_threshold_map[feature_names[feature_idx]] = sorted(
                    list(set(thresholds[thresholds != -2]))  # Remove leaf node markers
                )
                
    def get_model_params(self) -> Dict[str, Tuple[str, Any]]:
        """
        Define hyperparameter search space for Decision Tree.
        
        Returns:
            Dictionary of parameter names and their valid ranges/options
        """
        return {
            'max_depth': ('int', (3, 20)),
            'min_samples_split': ('int', (2, 20)),
            'min_samples_leaf': ('int', (1, 10)),
            'criterion': ('categorical', ['gini', 'entropy']),
            'ccp_alpha': ('float', (0.0, 0.05))
        }
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance based on the tree structure.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
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
        
    def get_decision_path(self, X: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        """
        Get the decision path for a single sample.
        
        Args:
            X: Single sample to analyze
            
        Returns:
            List of decision rules used for classification
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting decision path")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        X_processed = self.preprocess_data(X)
        
        feature_names = (self.feature_names if self.feature_names is not None 
                        else [f'feature_{i}' for i in range(self.model.n_features_in_)])
        
        # Get node indicator matrix
        node_indicator = self.model.decision_path(X_processed)
        
        # Get leaf ids
        leaf_id = self.model.apply(X_processed)
        
        # Get tree structure
        tree = self.model.tree_
        
        # Generate decision path
        path = []
        for sample_id in range(len(X)):
            # Get nodes for this sample
            node_index = node_indicator.indices[
                node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]
            ]
            
            for node_id in node_index[:-1]:  # Exclude leaf
                feature_id = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                
                if X_processed[sample_id, feature_id] <= threshold:
                    inequality = "<="
                else:
                    inequality = ">"
                    
                path.append(
                    f"{feature_names[feature_id]} {inequality} {threshold:.2f}"
                )
                
        return path
        
    def get_tree_visualization(self, 
                             max_depth: Optional[int] = None,
                             feature_names: Optional[List[str]] = None) -> str:
        """
        Generate a visualization of the decision tree.
        
        Args:
            max_depth: Maximum depth to show in visualization
            feature_names: Custom feature names to use
            
        Returns:
            Base64 encoded PNG image of the tree
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualization")
            
        plt.figure(figsize=(20, 10))
        
        if feature_names is None and self.feature_names is not None:
            feature_names = self.feature_names
            
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=list(self.label_encoder.classes_),
            filled=True,
            rounded=True,
            max_depth=max_depth
        )
        
        # Save plot to bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        image_png = buf.getvalue()
        buf.close()
        
        graphic = base64.b64encode(image_png).decode('utf-8')
        return graphic
        
    def get_feature_thresholds(self) -> Dict[str, List[float]]:
        """
        Get the threshold values used for each feature in the tree.
        
        Returns:
            Dictionary mapping feature names to their threshold values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting thresholds")
            
        return self.feature_threshold_map
        
    def cost_complexity_pruning(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Perform cost complexity pruning using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before pruning")
            
        # Get path of ccp_alphas
        path = self.model.cost_complexity_pruning_path(X_val, y_val)
        ccp_alphas = path.ccp_alphas
        
        # Create trees with different alphas
        trees = []
        for ccp_alpha in ccp_alphas:
            tree = DecisionTreeClassifier(
                random_state=42,
                ccp_alpha=ccp_alpha,
                **{k: v for k, v in self.model.get_params().items() if k != 'ccp_alpha'}
            )
            tree.fit(X_val, y_val)
            trees.append(tree)
            
        # Find best alpha
        accuracies = [tree.score(X_val, y_val) for tree in trees]
        best_alpha_idx = np.argmax(accuracies)
        
        # Update model with best alpha
        self.model.set_params(ccp_alpha=ccp_alphas[best_alpha_idx])
        self.model.fit(X_val, y_val)
        
        # Update thresholds
        self._store_feature_thresholds()