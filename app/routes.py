# app/routes.py

from flask import Blueprint, render_template, request, jsonify
import logging
from pathlib import Path
from flask import jsonify
import numpy as np
from typing import Dict, Any, Tuple
import pandas as pd

from .utils import (
    DataLoader,
    DataPreprocessor,
    ModelEvaluator,
    Visualizer,
    create_pipeline,
    load_and_prepare_data,
    get_initial_data_insights
)
from .models import get_model, list_available_models

# Create blueprint
home_bp = Blueprint('home', __name__, template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pipeline components
DATA_DIR = Path('data')
SAVE_DIR = Path('app/static')

try:
    data_loader, preprocessor, evaluator, visualizer = create_pipeline(
        str(DATA_DIR),
        str(SAVE_DIR)
    )
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {str(e)}")
    raise

def convert_to_serializable(obj):
    """
    Convert non-serializable objects (e.g., NumPy arrays, int64) to JSON-compatible formats.
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

@home_bp.route('/', methods=['GET'])
def home() -> str:
    """Landing page with settings form."""
    try:
        models = list_available_models()
        return render_template(
            "settings.html",
            available_models=models
        )
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        return render_template("error.html", error=str(e))

@home_bp.route('/analyze', methods=['GET'])
def analyze() -> str:
    """
    Process the analysis with selected settings.
    Handles data preparation, model training, evaluation, and visualization.
    Ensures all outputs are JSON serializable.
    """
    try:
        # Step 1: Parse request settings
        settings = {
            'processing_method': request.args.get('processing_method', 'none'),
            'model_name': request.args.get('model_name', 'Random Forest'),
            'use_hyperopt': request.args.get('use_hyperopt', 'false').lower() == 'true'
        }
        logger.info(f"Analysis requested with settings: {settings}")

        # Step 2: Load and split data
        try:
            df = data_loader.load_data("genres_v2.csv")
            X_train, X_test, y_train, y_test = data_loader.split_data(df)
            print(f"X_train shape before preprocessing: {X_train.shape}")
            print(f"X_test shape before preprocessing: {X_test.shape}")

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            print(f"X_train shape after preprocessing: {X_train_processed.shape}")
            print(f"X_test shape after preprocessing: {X_test_processed.shape}")
        except Exception as e:
            error_message = f"Data preparation failed: {str(e)}"
            logger.error(error_message)
            return jsonify({'error': error_message}), 500

        # Step 3: Generate insights
        try:
            insights = get_initial_data_insights(data_loader, visualizer, df)
        except Exception as e:
            error_message = f"Failed to generate insights: {str(e)}"
            logger.error(error_message)
            return jsonify({'error': error_message}), 500

        # Step 4: Train the model
        try:
            model = get_model(settings['model_name'])
            logger.info(f"Training {settings['model_name']}...")
            model.fit(X_train, y_train)
        except Exception as e:
            error_message = f"Model training failed: {str(e)}"
            logger.error(error_message)
            return jsonify({'error': error_message}), 500

        # Step 5: Evaluate the model and generate visualizations
        try:
            y_pred = model.predict(X_test)
            evaluation_results = evaluator.calculate_metrics(y_test, y_pred)

            # Generate confusion matrix plot
            conf_matrix_plot = visualizer.plot_confusion_matrix(
                y_test,
                y_pred,
                list(data_loader.get_genre_mapping().keys())
            )
        except Exception as e:
            error_message = f"Model evaluation failed: {str(e)}"
            logger.error(error_message)
            return jsonify({'error': error_message}), 500

        # Step 6: Create and return response
        response = {
            'success': True,
            'evaluation': convert_to_serializable(evaluation_results),
            'visualizations': {
                'confusion_matrix': conf_matrix_plot,
                'genre_distribution': insights['visualizations']['genre_distribution']
            }
        }
        return jsonify(response)

    except Exception as e:
        error_message = f"Analysis failed: {str(e)}"
        logger.error(error_message)
        return jsonify({'error': error_message}), 500

@home_bp.route('/api/models', methods=['GET'])
def get_available_models() -> Dict[str, Any]:
    """API endpoint to get available models."""
    try:
        return jsonify({
            'success': True,
            'models': list_available_models()
        })
    except Exception as e:
        logger.error(f"Failed to get models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@home_bp.route('/api/model_info/<model_name>', methods=['GET'])
def get_model_info(model_name: str) -> Dict[str, Any]:
    """API endpoint to get information about a specific model."""
    try:
        model = get_model(model_name)
        return jsonify({
            'success': True,
            'info': {
                'name': model_name,
                'parameters': model.get_model_params(),
                'description': model.__doc__
            }
        })
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@home_bp.errorhandler(404)
def not_found_error(error: Any) -> Tuple[str, int]:
    """Handle 404 errors."""
    logger.error(f"404 error: {error}")
    return render_template('error.html', error="Page not found"), 404

@home_bp.errorhandler(500)
def internal_error(error: Any) -> Tuple[str, int]:
    """Handle 500 errors."""
    logger.error(f"500 error: {error}")
    return render_template('error.html', error="Internal server error"), 500