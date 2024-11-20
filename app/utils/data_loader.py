# app/utils/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from category_encoders import TargetEncoder
from scipy import sparse
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_and_preprocess_data(filename, use_smote=False):
    """
    Loads and preprocesses data with advanced techniques to prevent overfitting
    """
    try:
        # Load data
        data_path = f"data/{filename}"
        df = pd.read_csv(data_path, low_memory=False)
        
        # Drop unwanted columns
        df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
        df = df.drop([
            "type", "id", "uri", "track_href", "analysis_url", "song_name", 
            "title"
        ], axis=1)
        
        # Remove duplicates
        original_size = len(df)
        df = df.drop_duplicates()
        print(f"Removed {original_size - len(df)} duplicate records")
        
        # Store original analysis
        from .analysis import analyze_dataset, check_data_quality
        original_analysis = analyze_dataset(df.copy())
        quality_checks = check_data_quality(df.copy())
        original_analysis['quality_checks'] = quality_checks
        
        # Feature Engineering
        df = engineer_features(df)
        
        # Separate features and target
        X = df.drop(['genre'], axis=1)
        y = df['genre']
        
        # Identify feature types
        categorical_features = X.select_dtypes(include=['object']).columns
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        
        # Create preprocessing pipeline with advanced imputation
        numeric_pipeline = Pipeline([
            ('imputer', IterativeImputer(random_state=42)),  # More sophisticated imputation
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),  # KNN-based imputation for categorical
            ('target_encoder', TargetEncoder(smoothing=2.0))  # Target encoding with smoothing
        ])
        
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Process numeric features
        X_train_numeric = numeric_pipeline.fit_transform(X_train[numeric_features])
        X_test_numeric = numeric_pipeline.transform(X_test[numeric_features])
        
        # Process categorical features
        if len(categorical_features) > 0:
            X_train_cat = categorical_pipeline.fit_transform(X_train[categorical_features], y_train)
            X_test_cat = categorical_pipeline.transform(X_test[categorical_features])
            
            # Combine features
            X_train_processed = np.hstack([X_train_numeric, X_train_cat])
            X_test_processed = np.hstack([X_test_numeric, X_test_cat])
        else:
            X_train_processed = X_train_numeric
            X_test_processed = X_test_numeric
        
        # Store distribution info
        original_analysis['original_class_distribution'] = y_train.value_counts().to_dict()
        
        # Apply SMOTE if enabled
        if use_smote:
            print("\nApplying SMOTE with advanced settings...")
            categorical_indices = np.arange(X_train_numeric.shape[1], X_train_processed.shape[1])
            
            smote = SMOTENC(
                categorical_features=categorical_indices,
                random_state=42,
                k_neighbors=min(5, min(y_train.value_counts())-1),
                sampling_strategy='not majority'  # More balanced approach
            )
            
            X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
            
            # Update distribution info
            original_analysis['smote_class_distribution'] = pd.Series(y_train).value_counts().to_dict()
            print("\nClass distribution after SMOTE:")
            print(pd.Series(y_train).value_counts())
        else:
            original_analysis['smote_class_distribution'] = y_train.value_counts().to_dict()
            
        original_analysis['smote_applied'] = use_smote
        
        return X_train_processed, X_test_processed, y_train, y_test, original_analysis
        
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {str(e)}")
        raise

def engineer_features(df):
    """
    Performs feature engineering to create more informative features
    """
    try:
        # Tempo-related features
        df['tempo_scaled'] = df['tempo'] / df['time_signature']
        df['tempo_energy_ratio'] = df['tempo'] / (df['energy'] + 1e-8)
        
        # Energy-related composite features
        df['energy_loudness_ratio'] = df['energy'] / (df['loudness'].abs() + 1e-8)
        df['energy_valence_product'] = df['energy'] * df['valence']
        
        # Acoustic features
        df['acoustic_energy_ratio'] = df['acousticness'] / (df['energy'] + 1e-8)
        df['acoustic_valence_ratio'] = df['acousticness'] / (df['valence'] + 1e-8)
        
        # Dance-related features
        df['dance_energy_ratio'] = df['danceability'] / (df['energy'] + 1e-8)
        df['dance_tempo_ratio'] = df['danceability'] / (df['tempo'] + 1e-8)
        
        # Normalize duration
        df['duration_normalized'] = (df['duration_ms'] - df['duration_ms'].mean()) / df['duration_ms'].std()
        
        # Create interaction terms
        df['speechiness_instrumental_ratio'] = df['speechiness'] / (df['instrumentalness'] + 1e-8)
        df['liveness_energy_product'] = df['liveness'] * df['energy']
        
        # Binned features
        df['tempo_bin'] = pd.qcut(df['tempo'], q=5, labels=['very_slow', 'slow', 'medium', 'fast', 'very_fast'])
        df['energy_bin'] = pd.qcut(df['energy'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        return df
        
    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        return df