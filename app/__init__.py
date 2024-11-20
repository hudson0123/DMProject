# app/__init__.py

import matplotlib
matplotlib.use('Agg')

from flask import Flask, current_app
import click
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Optional

def create_app(config_object: Optional[object] = None) -> Flask:
    """
    Flask application factory.
    
    Args:
        config_object: Configuration object (optional)
        
    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY') or 'dev-key-please-change',
        DATA_DIR=str(Path('data')),  # Convert to string for config
        STATIC_DIR=str(Path('app/static')),  # Convert to string for config
        MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
    )
    
    # Load additional configuration
    if config_object is not None:
        app.config.from_object(config_object)
        
    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
        
    # Ensure static and data directories exist
    Path(app.config['DATA_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app.config['STATIC_DIR']).mkdir(parents=True, exist_ok=True)
    
    # Register blueprints
    register_blueprints(app)
    
    return app

def register_blueprints(app: Flask) -> None:
    """
    Register Flask blueprints.
    
    Args:
        app: Flask application instance
    """
    # Import blueprints
    from .routes import home_bp
    
    # Register blueprints
    app.register_blueprint(home_bp)