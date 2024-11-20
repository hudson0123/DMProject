# app/utils/__init__.py

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from .data_loader import DataLoader
from .preprocessing import DataPreprocessor
from .evaluation import ModelEvaluator
from .visualization import Visualizer

__all__ = [
    'DataLoader',
    'DataPreprocessor',
    'ModelEvaluator',
    'Visualizer'
]

# Version of the utils package
__version__ = '1.0.0'

# Package metadata
__author__ = "Jackson Davis & Hudson O'Donnell"
__description__ = 'Utility modules for music genre classification'

def get_version() -> str:
    """Returns the version of the utils package."""
    return __version__

def create_pipeline(
    data_dir: str,
    save_dir: str = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataPreprocessor, ModelEvaluator, Visualizer]:
    """
    Create a complete data processing pipeline.
    
    Args:
        data_dir: Directory containing the dataset
        save_dir: Directory to save outputs (optional)
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (DataLoader, DataPreprocessor, ModelEvaluator, Visualizer)
    """
    data_loader = DataLoader(
        data_dir=data_dir,
        test_size=test_size,
        random_state=random_state
    )
    
    preprocessor = DataPreprocessor(
        scaler_type='standard',
        handle_outliers=True,
        handle_missing=True
    )
    
    evaluator = ModelEvaluator(save_path=save_dir)
    
    visualizer = Visualizer(save_dir=save_dir)
    
    return data_loader, preprocessor, evaluator, visualizer

def load_and_prepare_data(
    data_loader: DataLoader,
    preprocessor: DataPreprocessor,
    filename: str,
    consolidate_genres: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Load and prepare data using the provided components.
    
    Args:
        data_loader: Initialized DataLoader instance
        preprocessor: Initialized DataPreprocessor instance
        filename: Name of the data file
        consolidate_genres: Whether to consolidate rare genres
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    df = data_loader.load_data(filename)
    df = data_loader.prepare_data(df, consolidate_genres=consolidate_genres)
    X_train, X_test, y_train, y_test = data_loader.split_data(df)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test

def get_initial_data_insights(
    data_loader: DataLoader,
    visualizer: Visualizer,
    df: pd.DataFrame
) -> Dict:
    """
    Generate initial insights about the dataset.
    
    Args:
        data_loader: Initialized DataLoader instance
        visualizer: Initialized Visualizer instance
        df: Loaded DataFrame
        
    Returns:
        Dictionary containing data insights and visualizations
    """
    stats = data_loader.get_data_stats()
    
    genre_dist = visualizer.plot_genre_distribution(stats['genre_distribution'])
    feature_plots = visualizer.plot_feature_distributions(
        df, 
        data_loader.get_feature_names()
    )
    correlation_plot = visualizer.plot_correlation_matrix(
        df,
        data_loader.get_feature_names()
    )
    
    insights = {
        'statistics': stats,
        'visualizations': {
            'genre_distribution': genre_dist,
            'feature_distributions': feature_plots,
            'correlation_matrix': correlation_plot
        },
        'data_quality': data_loader.verify_data_quality(df)
    }
    
    return insights