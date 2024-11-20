import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_visualizations(df, stats):
    """
    Generates and saves visualizations.
    
    Parameters:
    df (pandas.DataFrame): Input dataset
    stats (dict): Statistics dictionary from analyze_dataset
    """
    try:
        # Create static directory if it doesn't exist
        if not os.path.exists('app/static'):
            os.makedirs('app/static')
        
        # Set style for all plots
        plt.style.use('seaborn')
        
        # Genre distribution plot
        create_genre_distribution(df)
        
        # Correlation heatmap
        create_correlation_heatmap(df, stats)
        
        # Feature distributions
        create_feature_distributions(df)
        
    except Exception as e:
        print(f"Error in save_visualizations: {str(e)}")
        plt.close('all')  # Make sure to close all figures in case of error
        raise

def create_feature_distributions(df):
    """Creates and saves individual feature distribution plots."""
    try:
        # Define the numeric features we want to plot
        features_to_plot = [
            'danceability', 'energy', 'key', 'loudness', 
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_ms'
        ]
        
        # Filter features that actually exist in the dataframe
        features_to_plot = [f for f in features_to_plot if f in df.columns]
        
        # Create directory for individual plots if it doesn't exist
        dist_dir = 'app/static/distributions'
        os.makedirs(dist_dir, exist_ok=True)
        
        # Create individual plots
        for col in features_to_plot:
            plt.figure(figsize=(10, 10))
            
            # Create the distribution plot with improved styling
            sns.histplot(data=df, x=col, kde=True, color='#4A90E2')
            
            # Customize the plot
            plt.title(col.replace('_', ' ').title(), fontsize=16, pad=20)
            plt.xlabel(col.replace('_', ' ').title(), fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.xticks(fontsize=12, rotation=45)
            plt.yticks(fontsize=12)
            
            # Add grid and style improvements
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            
            # Improve plot background
            plt.gca().set_facecolor('#f8f9fa')
            plt.gcf().set_facecolor('white')
            
            # Add padding to prevent label cutoff
            plt.tight_layout(pad=2.0)
            
            # Save individual plot with high resolution
            plt.savefig(f'{dist_dir}/{col}_dist.png', 
                       dpi=150,
                       bbox_inches='tight',
                       pad_inches=0.3,
                       facecolor='white')
            plt.close()
            
    except Exception as e:
        print(f"Error in create_feature_distributions: {str(e)}")
        plt.close('all')
        raise

def create_correlation_heatmap(df, stats):
    """Creates and saves the correlation heatmap."""
    try:
        plt.figure(figsize=(15, 12))  # Increased figure size
        
        # Ensure we exclude the Unnamed column and other non-numeric columns
        df_clean = df.select_dtypes(include=['float64', 'int64'])
        df_clean = df_clean.drop(['Unnamed: 0'], axis=1, errors='ignore')
        
        correlation_matrix = df_clean.corr()
        
        # Create heatmap with larger annotations
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   annot_kws={'size': 10})
        
        plt.title('Feature Correlations', pad=20)
        plt.tight_layout()
        plt.savefig('app/static/correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in create_correlation_heatmap: {str(e)}")
        plt.close('all')
        raise

def create_genre_distribution(df):
    """Creates and saves the genre distribution plot."""
    try:
        plt.figure(figsize=(15, 10))  # Increased figure size
        
        # Create countplot with rotated labels
        sns.countplot(data=df, 
                     y='genre', 
                     order=df['genre'].value_counts().index)
        
        plt.title('Genre Distribution', pad=20)
        plt.xlabel('Count')
        plt.ylabel('Genre')
        
        plt.tight_layout()
        plt.savefig('app/static/genre_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in create_genre_distribution: {str(e)}")
        plt.close('all')
        raise