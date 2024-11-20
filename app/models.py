# app/models.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Global configuration
enable_hyperparam_tuning = False

# Model definitions with optimized parameters
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=93,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    ),
    
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=3,
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

# Refined parameter ranges based on optimization results
param_ranges = {
    "Random Forest": {
        # Focus around successful values
        'n_estimators': ('int', (50, 300)),  # Narrowed around 93
        'max_depth': ('int', (7, 15)),       # Increased minimum, focused on higher values
        'min_samples_split': ('int', (2, 15)), # Narrowed around successful range
        'min_samples_leaf': ('int', (2, 15)),  # Narrowed around successful range
        'max_features': ('categorical', ['sqrt']),  # 'sqrt' consistently performed better
        'bootstrap': ('categorical', [True, False]),  # Added to try bagging variations
        'warm_start': ('categorical', [True, False]),  # Added for potential improvement
        'criterion': ('categorical', ['gini', 'entropy', 'log_loss'])  # Added to test different split criteria
    },
    
    "Gradient Boosting": {
        'n_estimators': ('int', (50, 250)),
        'max_depth': ('int', (2, 10)),
        'learning_rate': ('float', (0.001, 1)),
        'min_samples_split': ('int', (2, 15)),
        'min_samples_leaf': ('int', (1, 10)),
        'subsample': ('float', (0.3, 1.5))
    },
    
    "Logistic Regression": {
        'C': ('float', (0.001, 20.0)),
        'max_iter': ('int', (100, 5000))
    }
}

# Add advanced parameter controls
advanced_params = {
    "Random Forest": {
        'class_weight': 'balanced',
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }
}

# Function to update hyperparameter tuning setting
def set_hyperparam_tuning(enabled):
    global enable_hyperparam_tuning
    enable_hyperparam_tuning = enabled