# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
dataset = pd.read_csv(r'C:\Advanced projects\Pure Sckiet learn projects\Bangalore_traffic\Banglore_traffic_Dataset.csv')

# Step 2: Preprocess the 'Date' column
def preprocess_date_column(data):
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day
    data['day_of_week'] = data['Date'].dt.dayofweek
    return data

dataset = preprocess_date_column(dataset)

# Step 3: Define features and target variable
target_column = 'Traffic_Volume'
date_features = ['year', 'month', 'day', 'day_of_week']
categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Roadwork and Construction Activity']

X = dataset.drop(columns=[target_column])
y = dataset[target_column]

# Step 4: Preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), date_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_columns)  # Ensure dense output
    ]
)

# Step 5: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(random_state=42),
    'Bayesian Ridge': BayesianRidge()
}

# Step 7: Train and evaluate models
results = []
pipelines = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': model_name, 'MSE': mse, 'R²': r2})
    pipelines[model_name] = pipeline

# Convert results to a DataFrame for easier visualization
results_df = pd.DataFrame(results).sort_values(by='R²', ascending=False)

# Step 8: Visualize the results
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='R²', y='Model', palette='viridis')
plt.title('Model Performance (R² Score)')
plt.xlabel('R² Score')
plt.ylabel('Model')
plt.show()

# Step 9: Correlation heatmap
processed_columns = date_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns))
X_processed = preprocessor.transform(X)
X_processed_df = pd.DataFrame(X_processed, columns=processed_columns)
dataset_for_heatmap = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

corr_matrix = dataset_for_heatmap.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title("Correlation Heatmap")
plt.show()

# Step 10: Hyperparameter tuning (example for Random Forest)
param_grid_rf = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(
    Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))]),
    param_grid=param_grid_rf,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

print("Tuning Random Forest...")
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
best_rf_r2 = r2_score(y_test, best_rf.predict(X_test))
print(f"Best Random Forest Model R²: {best_rf_r2:.4f}")
