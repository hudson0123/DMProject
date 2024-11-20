# app/__init__.py
from flask import Flask
from config import Config

def create_app(config_class=Config):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions here if any
    
    # Register blueprints
    from .routes import home_bp
    app.register_blueprint(home_bp)
    
    return app

# Avoid circular imports by moving these to the end
from .utils.data_loader import load_and_preprocess_data
from .utils.model_utils import train_and_evaluate_models
from .utils.analysis import analyze_dataset, check_data_quality
from .utils.visualization import save_visualizations

__all__ = [
    'create_app',
    'load_and_preprocess_data',
    'train_and_evaluate_models',
    'analyze_dataset',
    'check_data_quality',
    'save_visualizations'
]