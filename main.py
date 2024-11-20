# main.py

import os
from pathlib import Path
from app import create_app

class Config:
    """Application configuration."""
    # Basic Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change'
    DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
    TESTING = False
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_EXTENSIONS = ['.csv']
    
    # Application paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = str(BASE_DIR / 'data')
    STATIC_DIR = str(BASE_DIR / 'app' / 'static')
    
    # Model configuration
    DEFAULT_MODEL = 'Random Forest'
    USE_HYPEROPT = False
    MIN_SAMPLES_PER_GENRE = 50
    
    # Other settings
    LOGGING_LEVEL = os.environ.get('LOGGING_LEVEL', 'INFO')

# Create application instance
app = create_app(Config)

if __name__ == '__main__':
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    app.run(
        host=host,
        port=port,
        debug=Config.DEBUG
    )