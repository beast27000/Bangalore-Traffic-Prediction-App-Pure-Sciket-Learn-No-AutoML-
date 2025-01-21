# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, DecisionTreeRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# STEP 1: Load and preprocess data
def load_and_preprocess_data(file_path):
    """Load dataset and preprocess date column."""
    # Load the dataset
    dataset = pd.read_csv(file_path)
    
    # Preprocess date column
    dataset = preprocess_date_column(dataset)
    
    # Prepare feature columns and target
    X = dataset.drop('Traffic_Volume', axis=1)  # Features
    y = dataset['Traffic_Volume']  # Target
    
    return X, y

def preprocess_date_column(data):
    """Preprocess the 'Date' column and extract date-related features."""
    if 'Date' not in data.columns:
        raise ValueError("The dataset does not contain a 'Date' column.")
    
    # Convert 'Date' to datetime and extract features
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    if data['Date'].isnull().any():
        print("Warning: Some dates could not be converted.")
    
    # Extract year, month, day, and day_of_week
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day
    data['day_of_week'] = data['Date'].dt.dayofweek
    
    return data

def preprocess_features(X):
    """Preprocess feature columns with scaling and encoding."""
    date_columns = ['year', 'month', 'day', 'day_of_week']
    categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Roadwork and Construction Activity']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), date_columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ]
    )
    
    return preprocessor.fit_transform(X)

# STEP 2: Model Training and Evaluation
def train_and_evaluate_models(X_train, y_train, X_test, y_test, models):
    """Train and evaluate multiple models, return best model."""
    pipelines = {}
    results = []

    # Loop through models, train and evaluate
    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        pipelines[model_name] = pipeline
        results.append((model_name, mse, r2))
        
        # Print evaluation
        print(f"{model_name} - MSE: {mse:.4f}, R²: {r2:.4f}")

    return pipelines, results

def select_best_model(pipelines, X_test, y_test):
    """Select the best model based on R² score."""
    best_model_name = max(pipelines, key=lambda model: r2_score(y_test, pipelines[model].predict(X_test)))
    best_model = pipelines[best_model_name]
    print(f"\nBest model: {best_model_name} with R²: {r2_score(y_test, best_model.predict(X_test)):.4f}")
    return best_model

# STEP 3: Hyperparameter Tuning
def tune_hyperparameters(pipelines, X_train, y_train, param_grids):
    """Perform GridSearchCV for hyperparameter tuning."""
    best_models = {}
    
    for model_name, pipeline in pipelines.items():
        print(f"Performing GridSearch for {model_name}...")
        grid_search = GridSearchCV(pipeline, param_grids.get(model_name, {}), cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        
    return best_models

# STEP 4: Cross-Validation
def cross_validate_models(models, X, y):
    """Perform cross-validation for each model."""
    cv_results = {}
    for model_name, model in models.items():
        print(f"Performing cross-validation for {model_name}...")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_results[model_name] = cv_scores
        print(f"Mean MSE for {model_name} (CV): {-cv_scores.mean():.4f}")
    return cv_results

# STEP 5: Feature Importance for Tree-based Models
def plot_feature_importance(models, X, preprocessor, date_columns):
    """Plot feature importances for tree-based models."""
    for model_name, model in models.items():
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            feature_importance = model.named_steps['model'].feature_importances_
            feature_names = date_columns + list(preprocessor.transformers_[1][1].get_feature_names_out())
            feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
            feature_df = feature_df.sort_values(by='Importance', ascending=False)
            print(f"Feature importance for {model_name}:")
            print(feature_df)

# STEP 6: Save Best Models
def save_best_models(best_models):
    """Save all the best models to disk."""
    for model_name, model in best_models.items():
        joblib.dump(model, f'{model_name}_best_model.pkl')
        print(f"Saved {model_name} to {model_name}_best_model.pkl")

# Main Execution
if __name__ == "__main__":
    file_path = 'path_to_your_data.csv'
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)
    X_processed = preprocess_features(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'SVR': SVR(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'XGBoost': XGBRegressor(),
        'Bayesian Ridge': BayesianRidge()
    }

    # Train and evaluate models
    pipelines, results = train_and_evaluate_models(X_train, y_train, X_test, y_test, models)
    
    # Select the best model
    best_model = select_best_model(pipelines, X_test, y_test)
    
    # Hyperparameter tuning
    param_grids = {
        'Random Forest': {'model__n_estimators': [100, 200]},
        'Decision Tree': {'model__max_depth': [None, 10]},
        # Add other model grids here...
    }
    best_models = tune_hyperparameters(pipelines, X_train, y_train, param_grids)
    
    # Cross-validation
    cv_results = cross_validate_models(best_models, X_processed, y)
    
    # Feature importance for tree models
    plot_feature_importance(best_models, X_test, preprocessor, ['year', 'month', 'day', 'day_of_week'])
    
    # Save the best models
    save_best_models(best_models)
