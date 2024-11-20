# app/models/__init__.py

"""
Model registry and factory pattern implementation for genre classification models.
Provides centralized model management and instantiation.
"""

from typing import Dict, Type, List
from .base_model import BaseGenreModel
from .naive_bayes import GaussianNBModel
from .random_forest import RandomForestModel
from .logistic_regression import LogisticRegressionModel
from .decision_tree import DecisionTreeModel

# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseGenreModel]] = {
    "Gaussian Naive Bayes": GaussianNBModel,
    "Random Forest": RandomForestModel,
    "Logistic Regression": LogisticRegressionModel,
    "Decision Tree": DecisionTreeModel
}

def get_model(model_name: str) -> BaseGenreModel:
    """
    Factory function to create model instances.
    
    Args:
        model_name (str): Name of the model to instantiate
        
    Returns:
        BaseGenreModel: Instance of the requested model
        
    Raises:
        ValueError: If model_name is not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list_available_models()}"
        )
    return MODEL_REGISTRY[model_name]()

def list_available_models() -> List[str]:
    """
    Returns list of available model names.
    
    Returns:
        List[str]: Names of available models
    """
    return list(MODEL_REGISTRY.keys())

def get_model_class(model_name: str) -> Type[BaseGenreModel]:
    """
    Get the model class without instantiating it.
    
    Args:
        model_name (str): Name of the model class to retrieve
        
    Returns:
        Type[BaseGenreModel]: The requested model class
        
    Raises:
        ValueError: If model_name is not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list_available_models()}"
        )
    return MODEL_REGISTRY[model_name]

# Default models to use if none specified
DEFAULT_MODELS = [
    "Random Forest",
    "Gaussian Naive Bayes",
    "Logistic Regression",
    "Decision Tree"
]