from flask import Blueprint, render_template, flash
from .utils import load_and_preprocess_data, train_and_evaluate_models

home_bp = Blueprint("home", __name__, template_folder="templates")

@home_bp.route("/")
def home():
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data("genres_v2.csv")
        
        # Train models and evaluate performance
        model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error: {e}")
        model_results = {}
        flash(f"An error occurred: {e}", "error")
    
    # Render the home template with model results
    return render_template("home.html", model_results=model_results)

