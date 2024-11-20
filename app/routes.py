# app/routes.py
from flask import Blueprint, render_template, flash, request
from .utils import load_and_preprocess_data, train_and_evaluate_models
from .models import set_hyperparam_tuning
import sys

home_bp = Blueprint("home", __name__, template_folder="templates")

@home_bp.route("/", methods=['GET'])
def home():
    """Landing page with settings form"""
    return render_template("settings.html")

@home_bp.route("/analyze", methods=['GET'])
def analyze():
    """Process the analysis with selected settings"""
    try:
        # Get settings from query parameters
        use_smote = request.args.get('use_smote', 'false').lower() == 'true'
        use_hyperopt = request.args.get('use_hyperopt', 'false').lower() == 'true'
        
        # Update hyperparameter tuning setting
        set_hyperparam_tuning(use_hyperopt)
        
        print(f"\nSettings applied:")
        print(f"SMOTE: {use_smote}")
        print(f"Hyperparameter Tuning: {use_hyperopt}")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, analysis_results = load_and_preprocess_data(
            "genres_v2.csv",
            use_smote=use_smote
        )
        
        # Train models and evaluate performance
        model_results = train_and_evaluate_models(
            X_train, X_test, y_train, y_test, 
            use_smote=use_smote
        )
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        model_results = {}
        analysis_results = {}
        flash(f"An error occurred: {str(e)}", "error")
    
    return render_template(
        "results.html",
        model_results=model_results,
        analysis_results=analysis_results,
        use_smote=use_smote,
        use_hyperopt=use_hyperopt
    )