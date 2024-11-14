
# Music Genre Classification API

This application classifies music genres using Spotify API data. It runs four classifiers—Multinomial Naive Bayes, Random Forest, Support Vector Classifier, and Logistic Regression—and evaluates model performance using various metrics. Hyperparameter tuning can be enabled or disabled as needed.

## Getting Started

Follow these steps to set up and run the application locally using Docker.

### Prerequisites
- Docker installed on your machine.

### Installation and Setup

1. Build the Docker Image
   ```
   docker build . -t <ImageName>
   ```
   Replace `<ImageName>` with your preferred name for the Docker image.

2. Run the Docker Container
   ```
   docker run -p 5000:5000 <ImageName>
   ```
   This will start the Flask application on port `5000`.

3. Access the Application
   - Open your browser and go to: http://localhost:5000. Give the application time to load the data and train the model. This will take much longer with hyperparameter tuning set at True and additional models added to the models dictionary.

## Model Details

The application currently supports the following classifiers:
- Multinomial Naive Bayes
- Random Forest (currently configured with hyperparameter tuning enabled by default)
- Support Vector Classifier (SVC)
- Logistic Regression

### Hyperparameter Tuning
- Hyperparameter tuning is currently disabled for Random Forest.
- To enable or disable tuning for specific models, adjust the `enable_hyperparam_tuning` variable in the code.

## Evaluation Techniques

The following evaluation metrics are used to assess model performance:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Cohen's Kappa
- Hamming Loss
- Classification Report

These metrics are displayed on the application page once the models have been trained and evaluated.

## Authors
- Hudson O'Donnell
- Jackson Davis