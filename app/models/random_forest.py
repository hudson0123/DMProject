# app/models/random_forest.py

from typing import Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from .base_model import BaseGenreModel

class RandomForestModel(BaseGenreModel):
    """
    Random Forest implementation for genre classification.
    Includes feature importance-based selection and specific preprocessing.
    
    Features:
    - Automatic feature selection based on importance
    - Feature interaction generation
    - Handling of class imbalance
    - Comprehensive feature importance analysis
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        importance_threshold: float = 0.01
    ):
        """
        Initialize Random Forest model with custom configurations.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at each leaf node
            importance_threshold: Minimum feature importance to keep feature
        """
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.importance_threshold = importance_threshold
        self.feature_selector: Optional[SelectFromModel] = None
        self.selected_feature_mask: Optional[np.ndarray] = None
        self.interaction_features: Optional[list] = None
        
    def _generate_interactions(self, X: np.ndarray) -> np.ndarray:
        """
        Generate interaction features between important features.
        
        Args:
            X: Input features
            
        Returns:
            Array with original and interaction features
        """
        if not self.is_fitted or self.selected_feature_mask is None:
            return X
            
        # Get indices of important features
        important_indices = np.where(self.selected_feature_mask)[0]
        
        # Generate interactions only if we have multiple important features
        if len(important_indices) > 1:
            interactions = []
            self.interaction_features = []  # Store interaction feature names
            
            # Generate pairwise interactions for important features
            for i in range(len(important_indices)):
                for j in range(i + 1, len(important_indices)):
                    idx1, idx2 = important_indices[i], important_indices[j]
                    interaction = X[:, idx1] * X[:, idx2]
                    interactions.append(interaction.reshape(-1, 1))
                    
                    # Store interaction feature names
                    if self.feature_names is not None:
                        self.interaction_features.append(
                            f"{self.feature_names[idx1]}_{self.feature_names[idx2]}_interaction"
                        )
                    else:
                        self.interaction_features.append(
                            f"feature_{idx1}_feature_{idx2}_interaction"
                        )
            
            if interactions:
                return np.hstack([X] + interactions)
        
        return X
        
    def preprocess_data(self, X: Union[np.ndarray, pd.DataFrame],
                    y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Preprocess data specifically for Random Forest:
        1. Scale features
        2. Select important features
        3. Generate interaction features
        
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
            
            # Initial fit to determine feature importances
            self.model.fit(X_scaled, y)
            
            # Create feature selector based on importance threshold
            self.feature_selector = SelectFromModel(
                self.model,
                threshold=self.importance_threshold,
                prefit=True
            )
            self.selected_feature_mask = self.feature_selector.get_support()
            
            # Select important features
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Generate interaction features
            X_interactions = self._generate_interactions(X_selected)

            print(f"X_selected shape: {X_selected.shape}")
            print(f"X_interactions shape: {X_interactions.shape}")
            
            return X_interactions
            
        # During prediction
        else:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Select important features if selection was done during training
            if self.feature_selector is not None:
                X_selected = self.feature_selector.transform(X_scaled)
                # Generate interaction features
                X_interactions = self._generate_interactions(X_selected)
                return X_interactions
                
            return X_scaled
            
    def get_model_params(self) -> Dict[str, Tuple[str, Any]]:
        """
        Define hyperparameter search space for Random Forest.
        
        Returns:
            Dictionary of parameter names and their valid ranges/options
        """
        return {
            'n_estimators': ('int', (100, 500)),
            'max_depth': ('int', (10, 100)),
            'min_samples_split': ('int', (2, 20)),
            'min_samples_leaf': ('int', (1, 10)),
            'importance_threshold': ('float', (0.001, 0.1))
        }
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores including interaction features.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return None
            
        importances = self.model.feature_importances_
        
        # Get base feature names
        if self.feature_names is None:
            base_features = [f'feature_{i}' for i in range(len(self.selected_feature_mask))]
        else:
            base_features = self.feature_names
            
        # Get selected feature names
        selected_features = [
            name for name, selected in zip(base_features, self.selected_feature_mask)
            if selected
        ]
        
        # Add interaction feature names if they exist
        if self.interaction_features:
            all_features = selected_features + self.interaction_features
        else:
            all_features = selected_features
            
        # Create importance dictionary
        importance_dict = dict(zip(all_features, importances))
        
        # Sort by importance
        return dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
    def get_feature_interactions(self) -> Optional[Dict[str, float]]:
        """
        Get the strength of feature interactions based on their importance.
        
        Returns:
            Dictionary mapping interaction names to their importance scores
        """
        if not self.is_fitted or not self.interaction_features:
            return None
            
        importances = self.model.feature_importances_
        n_original = sum(self.selected_feature_mask)
        interaction_importances = importances[n_original:]
        
        return dict(sorted(
            zip(self.interaction_features, interaction_importances),
            key=lambda x: x[1],
            reverse=True
        ))