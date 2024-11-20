# Configure matplotlib first
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_dataset(df):
    """
    Performs comprehensive analysis of the dataset.
    
    Parameters:
    df (pandas.DataFrame): Input dataset
    
    Returns:
    dict: Dictionary containing analysis results
    """
    try:
        # Basic dataset statistics
        stats = {
            "total_records": len(df),
            "features": len(df.columns),
            "genres": df['genre'].nunique(),
            "genre_distribution": df['genre'].value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "feature_stats": df.describe().to_dict()
        }
        
        # Check for class imbalance
        genre_counts = df['genre'].value_counts()
        stats["imbalance_ratio"] = genre_counts.max() / genre_counts.min()
        
        # Feature correlations for numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        stats["correlations"] = df[numerical_cols].corr()
        
        # Generate and save visualizations
        from .visualization import save_visualizations
        save_visualizations(df, stats)
        
        return stats
        
    except Exception as e:
        print(f"Error in analyze_dataset: {str(e)}")
        raise

def check_data_quality(df):
    """
    Performs data quality checks.
    
    Parameters:
    df (pandas.DataFrame): Input dataset
    
    Returns:
    dict: Dictionary containing quality check results
    """
    try:
        quality_checks = {
            "duplicates": df.duplicated().sum(),
            "missing_values": df.isnull().sum().to_dict(),
            "feature_ranges": {
                col: {
                    "min": df[col].min(), 
                    "max": df[col].max()
                } 
                for col in df.select_dtypes(include=['float64', 'int64']).columns
            },
            "outliers": {}
        }
        
        # Check for outliers using IQR method
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            quality_checks["outliers"][col] = outliers
        
        return quality_checks
        
    except Exception as e:
        print(f"Error in check_data_quality: {str(e)}")
        raise