# app/utils/model_utils.py
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, cohen_kappa_score, hamming_loss,
    confusion_matrix, balanced_accuracy_score
)
import numpy as np
from sklearn.base import clone
import optuna
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

def create_error_metrics():
    """Creates a dictionary of error metrics"""
    return {
        "model": None,
        "accuracy": 0.0,
        "precision_weighted": 0.0,
        "recall_weighted": 0.0,
        "f1_weighted": 0.0,
        "cohen_kappa": 0.0,
        "hamming_loss": 1.0,
        "classification_report": {"error": "Model training failed"},
        "confusion_matrix": np.zeros((1, 1)),
        "cv_score": 0.0,
        "cv_std": 0.0
    }

def create_objective(X_train, y_train, model_class, param_ranges, cv=5):
    """Creates an objective function with faster training strategies"""
    def objective(trial):
        params = {}
        
        # Add trial parameters
        for param, (param_type, param_range) in param_ranges.items():
            if param_type == 'int':
                params[param] = trial.suggest_int(param, param_range[0], param_range[1])
            elif param_type == 'float':
                params[param] = trial.suggest_float(param, param_range[0], param_range[1], log=True)
            elif param_type == 'categorical':
                params[param] = trial.suggest_categorical(param, param_range)
        
        # Add default params for specific models
        if model_class.__name__ == "GradientBoostingClassifier":
            params.update({
                'validation_fraction': 0.1,
                'n_iter_no_change': 5,
                'tol': 1e-4
            })
        
        model = model_class(**params)
        
        # Use different evaluation strategies based on the model
        if model_class.__name__ == "GradientBoostingClassifier":
            # Use a single validation split for faster evaluation
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            model.fit(X_train_split, y_train_split)
            score = balanced_accuracy_score(y_val_split, model.predict(X_val_split))
        else:
            # Regular cross-validation for other models
            scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring='balanced_accuracy',
                n_jobs=-1
            )
            score = scores.mean()
        
        return score
    
    return objective

def optimize_hyperparameters(X_train, y_train, model_class, param_ranges, n_trials=200):
    """Performs hyperparameter optimization with early stopping"""
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=3
        )
    )
    
    objective = create_objective(X_train, y_train, model_class, param_ranges)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value

def train_and_evaluate_models(X_train, X_test, y_train, y_test, use_smote=False):
    """Trains and evaluates models with anti-overfitting measures"""
    try:
        from ..models import models, param_ranges
        results = {}
        trained_models = {}
        
        # Process each model
        for model_name, base_model in models.items():
            try:
                print(f"\nTraining {model_name}...")
                
                # Clone model
                model = clone(base_model)
                
                # Optimize hyperparameters if available
                if model_name in param_ranges:
                    best_params, best_score = optimize_hyperparameters(
                        X_train, y_train,
                        type(model),
                        param_ranges[model_name]
                    )
                    print(f"Best parameters: {best_params}")
                    print(f"Best CV score: {best_score:.3f}")
                    model.set_params(**best_params)
                
                # Train model
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = calculate_metrics(model, X_test, y_test, y_pred)
                metrics['cv_score'] = best_score if model_name in param_ranges else 0.0
                
                results[model_name] = metrics
                
            except Exception as model_error:
                print(f"Error training {model_name}: {str(model_error)}")
                results[model_name] = create_error_metrics()
        
        # Create and evaluate voting ensemble
        try:
            if trained_models:
                print("\nTraining Voting Ensemble...")
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in trained_models.items()],
                    voting='soft'
                )
                ensemble.fit(X_train, y_train)
                y_pred_ensemble = ensemble.predict(X_test)
                ensemble_metrics = calculate_metrics(ensemble, X_test, y_test, y_pred_ensemble)
                results['Voting Ensemble'] = ensemble_metrics
                
        except Exception as ensemble_error:
            print(f"Error training ensemble: {str(ensemble_error)}")
        
        return results
        
    except Exception as e:
        print(f"Error in train_and_evaluate_models: {str(e)}")
        return {"Error": create_error_metrics()}

def calculate_metrics(model, X_test, y_test, y_pred):
    """Calculates evaluation metrics"""
    try:
        metrics = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average='weighted'),
            "recall_weighted": recall_score(y_test, y_pred, average='weighted'),
            "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
            "cohen_kappa": cohen_kappa_score(y_test, y_pred),
            "hamming_loss": hamming_loss(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        return metrics
    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        return create_error_metrics()