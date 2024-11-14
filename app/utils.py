from .models import models, param_grids

# Import statements for Flask and the main application
from flask import Flask, Blueprint, render_template, flash

# Importing libraries for data handling and model evaluation
import numpy as np
import pandas as pd

# Scikit-learn modules for machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree

# Scikit-learn modules for data preprocessing and model evaluation
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    cohen_kappa_score,
    hamming_loss
)

enable_hyperparam_tuning = False

def load_and_preprocess_data(filename):
    """
    Loads data from a CSV file, preprocesses it, and splits it into training and testing sets.
    
    Parameters:
    filename (str): The path to the CSV file containing the dataset.
    
    Returns:
    tuple: A tuple containing the training and testing data: (X_train, X_test, y_train, y_test).
    """
    # Load data
    data_path = f"data/{filename}"
    df = pd.read_csv(data_path)
    
    # Drop irrelevant columns
    df = df.drop(["type", "id", "uri", "track_href", "analysis_url", "song_name", "Unnamed: 0", "title", "duration_ms", "time_signature", "mode"], axis=1)
    
    # Separate features and target
    X = df.drop(['genre'], axis=1)
    y = df['genre']
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Preprocessing pipeline for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with the median
        ('scaler', MinMaxScaler())  # Scale values between 0 and 1
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with the most frequent value
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Transform data
    X_processed = preprocessor.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name} with {'tuning' if enable_hyperparam_tuning else 'default parameters'}...")
        
        if enable_hyperparam_tuning:
            tuned_model = tune_and_train_model(model, param_grids[model_name], X_train, y_train)
        else:
            model.fit(X_train, y_train)
            tuned_model = model
            
        y_pred = tuned_model.predict(X_test)
        
        results[model_name] = {
            "best_params": tuned_model.get_params() if enable_hyperparam_tuning else "Default parameters",
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "cohen_kappa": cohen_kappa_score(y_test, y_pred),
            "hamming_loss": hamming_loss(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }
    
    return results
