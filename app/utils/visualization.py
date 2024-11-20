# app/utils/visualization.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import io
import base64
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Handles visualization generation for music genre classification.
    Creates and saves various plots for data analysis and model evaluation.
    """
    
    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations (optional)
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
        # Set style
        plt.style.use('seaborn')
        
    def _save_plot(self, name: str) -> str:
        """
        Save plot to file and return base64 string.
        
        Args:
            name: Name of the plot
            
        Returns:
            Base64 encoded string of the plot
        """
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        
        # Save to file if directory specified
        if self.save_dir:
            plt.savefig(self.save_dir / f"{name}.png", dpi=300, bbox_inches='tight')
            
        return img_str
        
    def plot_genre_distribution(self, genre_counts: Dict[str, int]) -> str:
        """
        Create bar plot of genre distribution.

        Args:
            genre_counts: Dictionary of genre counts

        Returns:
            Base64 encoded plot image or an empty string if data is unavailable.
        """
        if not genre_counts:
            logger.warning("No data provided for genre distribution plot.")
            return ""  # Return empty string for no-data scenarios

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(genre_counts.values()), y=list(genre_counts.keys()))
        plt.title('Genre Distribution')
        plt.xlabel('Number of Samples')
        plt.ylabel('Genre')

        return self._save_plot('genre_distribution')

        
    def plot_feature_distributions(self, df: pd.DataFrame, 
                                 features: List[str]) -> Dict[str, str]:
        """
        Create distribution plots for features.
        
        Args:
            df: DataFrame containing features
            features: List of feature names to plot
            
        Returns:
            Dictionary mapping feature names to base64 encoded plots
        """
        plots = {}
        
        for feature in features:
            plt.figure(figsize=(8, 6))
            
            # Create distribution plot
            sns.histplot(data=df, x=feature, kde=True)
            
            plt.title(f'{feature} Distribution')
            plt.xlabel(feature)
            plt.ylabel('Count')
            
            plots[feature] = self._save_plot(f'{feature}_distribution')
            
        return plots
        
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                              features: List[str]) -> str:
        """
        Create correlation matrix heatmap.
        
        Args:
            df: DataFrame containing features
            features: List of features to include
            
        Returns:
            Base64 encoded plot image
        """
        plt.figure(figsize=(12, 10))
        
        # Calculate correlations
        corr_matrix = df[features].corr()
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        
        plt.title('Feature Correlations')
        
        return self._save_plot('correlation_matrix')
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: List[str]) -> str:
        """
        Create confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            
        Returns:
            Base64 encoded plot image
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        return self._save_plot('confusion_matrix')
        
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              top_n: int = 10) -> str:
        """
        Create feature importance bar plot.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            top_n: Number of top features to show
            
        Returns:
            Base64 encoded plot image
        """
        plt.figure(figsize=(10, 6))
        
        # Sort and select top features
        sorted_features = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n])
        
        # Create bar plot
        sns.barplot(
            x=list(sorted_features.values()),
            y=list(sorted_features.keys())
        )
        
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        
        return self._save_plot('feature_importance')
        
    def plot_dimension_reduction(self, X: np.ndarray, y: np.ndarray,
                               method: str = 'pca') -> str:
        """
        Create 2D visualization of high-dimensional data.
        
        Args:
            X: Feature matrix
            y: Labels
            method: Dimension reduction method ('pca' or 'tsne')
            
        Returns:
            Base64 encoded plot image
        """
        plt.figure(figsize=(10, 8))
        
        # Perform dimension reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, random_state=42)
            
        X_reduced = reducer.fit_transform(X)
        
        # Create scatter plot
        scatter = plt.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=y,
            cmap='tab20',
            alpha=0.6
        )
        
        plt.title(f'{method.upper()} Visualization')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.legend(
            scatter.legend_elements()[0],
            np.unique(y),
            title='Genres',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        return self._save_plot(f'{method}_visualization')
        
    def plot_learning_curves(self, train_scores: List[float], 
                           val_scores: List[float], 
                           train_sizes: List[int]) -> str:
        """
        Create learning curve plot.
        
        Args:
            train_scores: List of training scores
            val_scores: List of validation scores
            train_sizes: List of training set sizes
            
        Returns:
            Base64 encoded plot image
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_scores, label='Training Score', marker='o')
        plt.plot(train_sizes, val_scores, label='Validation Score', marker='o')
        
        plt.title('Learning Curves')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        return self._save_plot('learning_curves')
        
    def create_dashboard(self, data_stats: Dict, model_results: Dict) -> str:
        """
        Create comprehensive visualization dashboard in HTML.
        
        Args:
            data_stats: Data statistics dictionary
            model_results: Model evaluation results
            
        Returns:
            HTML string containing dashboard
        """
        html = """
        <div class="dashboard">
            <style>
                .dashboard {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 20px;
                    border-radius: 5px;
                    background-color: #f8f9fa;
                }
                .plot-container {
                    margin: 20px 0;
                    text-align: center;
                }
                img {
                    max-width: 100%;
                    height: auto;
                }
            </style>
        """
        
        # Add data statistics section
        html += """
            <div class="section">
                <h2>Dataset Statistics</h2>
                <ul>
        """
        
        for key, value in data_stats.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
            
        html += "</ul></div>"
        
        # Add plots
        for name, plot in model_results.items():
            if isinstance(plot, str) and plot.startswith('data:image'):
                html += f"""
                    <div class="section">
                        <h2>{name}</h2>
                        <div class="plot-container">
                            <img src="{plot}" alt="{name}">
                        </div>
                    </div>
                """
                
        html += "</div>"
        
        return html