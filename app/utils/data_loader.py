# app/utils/data_loader.py

from typing import Dict, Tuple, Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and initial preparation of music genre data.
    Provides functionality for data loading, validation, and splitting.
    """
    
    # Columns that should be dropped as they're not useful for prediction
    COLUMNS_TO_DROP = [
        "type", "id", "uri", "track_href", "analysis_url",
        "song_name", "title", "Unnamed: 0"
    ]
    
    # Required columns for genre classification
    REQUIRED_COLUMNS = [
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "duration_ms",
        "time_signature", "genre"
    ]
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        test_size: float = 0.2,
        random_state: int = 42,
        min_samples_per_genre: int = 50
    ):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the dataset
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            min_samples_per_genre: Minimum samples required per genre
        """
        self.data_dir = Path(data_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.min_samples_per_genre = min_samples_per_genre
        
        # State tracking
        self.data_stats: Dict = {}
        self.genre_mapping: Dict[str, int] = {}
        self.feature_names: List[str] = []
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from file and perform initial cleaning.

        Args:
            filename: Name of the data file

        Returns:
            Cleaned DataFrame

        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load data
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Loaded {len(df)} rows from {filename}")

        # Validate required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove unnecessary columns
        cols_to_drop = [col for col in self.COLUMNS_TO_DROP if col in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Store feature names
        self.feature_names = [col for col in df.columns if col != 'genre']

        # Calculate and store data statistics
        self._calculate_data_stats(df)

        return df


    def _calculate_data_stats(self, df: pd.DataFrame) -> None:
        """
        Calculate and store various statistics about the dataset.

        Args:
            df: Input DataFrame
        """
        self.data_stats = {
            'total_samples': len(df),
            'num_features': len(self.feature_names),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(exclude=[np.number]).columns.tolist(),
        }

        # Ensure genre distribution is calculated for training insights
        if 'genre' in df.columns and not df['genre'].isnull().all():
            self.data_stats['genre_distribution'] = df['genre'].value_counts().to_dict()
        else:
            logger.warning("The 'genre' column is missing or contains no valid data.")
            self.data_stats['genre_distribution'] = {}

        logger.info(f"Data statistics calculated: {self.data_stats}")




        
    def prepare_data(self, df: pd.DataFrame, consolidate_genres: bool = True) -> pd.DataFrame:
        """
        Prepare data for training by handling edge cases and optionally consolidating genres.
        
        Args:
            df: Input DataFrame
            consolidate_genres: Whether to consolidate rare genres into 'Other'
            
        Returns:
            Prepared DataFrame
        """
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Consolidate genres if requested
        if consolidate_genres:
            df = self._consolidate_rare_genres(df)
            
        # Create genre mapping
        unique_genres = df['genre'].unique()
        self.genre_mapping = {genre: idx for idx, genre in enumerate(sorted(unique_genres))}
        
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        numeric_features = df.select_dtypes(include=[np.number]).columns
        
        # For numeric features, fill missing values with median
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
        
        # For categorical features (including genre), fill with mode
        categorical_features = df.select_dtypes(exclude=[np.number]).columns
        df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])
        
        return df
        
    def _consolidate_rare_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Consolidate genres with few samples into an 'Other' category.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with consolidated genres
        """
        genre_counts = df['genre'].value_counts()
        rare_genres = genre_counts[genre_counts < self.min_samples_per_genre].index
        
        if not rare_genres.empty:
            logger.info(f"Consolidating {len(rare_genres)} rare genres into 'Other'")
            df.loc[df['genre'].isin(rare_genres), 'genre'] = 'Other'
            
        return df
        
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Ensure 'genre' column is present for splitting
        if 'genre' not in df.columns:
            raise ValueError("The dataset must contain a 'genre' column for splitting.")

        X = df.drop('genre', axis=1)
        y = df['genre']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        logger.info(f"Split data into {len(X_train)} training and {len(X_test)} test samples")

        return X_train, X_test, y_train, y_test


    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
        
    def get_genre_mapping(self) -> Dict[str, int]:
        """Get mapping of genres to indices."""
        return self.genre_mapping
        
    def get_data_stats(self) -> Dict:
        """Get statistics about the loaded data."""
        return self.data_stats
        
    def verify_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform quality checks on the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing quality metrics
        """
        quality_metrics = {
            'duplicates': df.duplicated().sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'feature_ranges': {
                feature: {
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'mean': df[feature].mean(),
                    'std': df[feature].std()
                }
                for feature in self.feature_names
                if df[feature].dtype in [np.float64, np.int64]
            }
        }
        
        return quality_metrics