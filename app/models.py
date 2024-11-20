# app/models.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Global configuration
enable_hyperparam_tuning = False

# Model definitions with conservative parameters
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    ),
    
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,  # Reduced for faster training
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    ),
    
    "Logistic Regression": LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
}

# Parameter ranges for Optuna optimization
param_ranges = {
    "Random Forest": {
        'n_estimators': ('int', (50, 200)),
        'max_depth': ('int', (3, 8)),
        'min_samples_split': ('int', (5, 15)),
        'min_samples_leaf': ('int', (3, 10)),
        'max_features': ('categorical', ['sqrt', 'log2'])
    },
    
    "Gradient Boosting": {
        'n_estimators': ('int', (30, 80)),
        'max_depth': ('int', (2, 4)),
        'learning_rate': ('float', (0.05, 0.3)),
        'min_samples_split': ('int', (10, 30)),
        'min_samples_leaf': ('int', (5, 15)),
        'subsample': ('float', (0.6, 0.8))
    },
    
    "Logistic Regression": {
        'C': ('float', (0.01, 10.0)),
        'max_iter': ('int', (500, 2000))
    }
}

def set_hyperparam_tuning(enabled):
    """Updates the hyperparameter tuning setting"""
    global enable_hyperparam_tuning
    enable_hyperparam_tuning = enabled