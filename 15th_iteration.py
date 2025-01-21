import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Paths for saving models and encoders
MODEL_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\traffic_regression_model.pkl"
ENCODER_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\label_encoders.pkl"

# -------------------- DATA LOADING AND PREPROCESSING --------------------

# Load the dataset
data = pd.read_csv("C:/Advanced projects/Bangalore_Traffic/Banglore_traffic_Dataset.csv")

# Convert 'Date' column to datetime format and extract date-related features
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.weekday

# Handle categorical columns by label encoding them
categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Traffic Signal Compliance', 'Roadwork and Construction Activity']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    # Fill missing values before encoding to avoid errors
    data[col] = le.fit_transform(data[col].fillna('missing'))
    encoders[col] = le

# Save the encoders to disk for future use in predictions
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoders, f)
print(f"Label encoders saved to {ENCODER_PATH}")

# Fill missing numeric values with mean and categorical values with mode
data = data.fillna(data.mean())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# -------------------- DATA SPLITTING AND FEATURE SCALING --------------------

# Splitting the dataset into features (X) and target (y)
X = data.drop(columns=['Traffic_Volume'])
y = data['Traffic_Volume']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Feature scaling for numeric columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
numeric_transformer = StandardScaler()

# Preprocessing pipeline: Scaling numeric columns and passing categorical columns without transformation
preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', numeric_transformer, numeric_features),
        ('cat', 'passthrough', categorical_columns)  # No transformation for categorical columns
    ])

# -------------------- MODEL SELECTION --------------------

# Define different regression models to test
models = {
    'Random Forest': RandomForestRegressor(random_state=123),
    'Linear Regression': LinearRegression(),
    'Support Vector Regressor': SVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=123),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Elastic Net Regression': ElasticNet(alpha=1.0, l1_ratio=0.5),
    'Bayesian Ridge Regression': BayesianRidge(),
    'XGBoost': XGBRegressor(random_state=123),
    'LightGBM': LGBMRegressor(random_state=123),
    'CatBoost': CatBoostRegressor(random_state=123, verbose=0)
}

# -------------------- MODEL TRAINING AND EVALUATION --------------------

# Model performance tracking
model_performance = []
best_score = -float('inf')
best_model = None
best_model_name = None

# Evaluate all models
for name, model in models.items():
    pipeline = Pipeline(steps=[ 
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)

    # Predict and evaluate performance on the test set
    y_pred = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    model_performance.append((name, test_r2))

    # Track the best performing model
    if test_r2 > best_score:
        best_score = test_r2
        best_model = pipeline
        best_model_name = name

# Save the best model to a file for later use
with open(MODEL_PATH, 'wb') as model_file:
    pickle.dump(best_model, model_file)
print(f"Best model saved to {MODEL_PATH}")

# -------------------- FINAL MODEL EVALUATION --------------------

# Evaluate the best model on the test set
y_pred_best = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_best)
test_r2 = r2_score(y_test, y_pred_best)

# Calculate Adjusted R²
n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)

# Calculate RMSE
rmse = np.sqrt(test_mse)

# -------------------- STREAMLIT APP FOR PREDICTIONS AND VISUALIZATIONS --------------------

# Streamlit App for Predictions and Performance Metrics
st.title(f"Bangalore Traffic Prediction App - Best Model: {best_model_name}")

# -------------------- SECTION 1: TRAINING DATA HEATMAP AND PERFORMANCE --------------------

st.header("Training Data Visualizations")

# Heatmap for training data correlation
st.subheader("Training Data Heatmap")
train_corr_matrix = X_train.copy()
train_corr_matrix['Traffic_Volume'] = y_train
train_corr = train_corr_matrix.corr()  # Compute correlation matrix for train set
plt.figure(figsize=(12, 8))
sns.heatmap(train_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
st.pyplot(plt)

# Performance Metrics for training data
st.subheader("Model Performance on Training Data")
train_y_pred = best_model.predict(X_train)
train_mse = mean_squared_error(y_train, train_y_pred)
train_r2 = r2_score(y_train, train_y_pred)
train_rmse = np.sqrt(train_mse)
train_adj_r2 = 1 - (1 - train_r2) * (X_train.shape[0] - 1) / (X_train.shape[0] - X_train.shape[1] - 1)

# Display metrics
st.write(f"Mean Squared Error (MSE) on Training Data: {train_mse}")
st.write(f"R² Score on Training Data: {train_r2}")
st.write(f"Root Mean Squared Error (RMSE) on Training Data: {train_rmse}")
st.write(f"Adjusted R² on Training Data: {train_adj_r2}")

# Bar Graph for Model Performance Comparison (Updated for better view)
st.subheader("Model Performance Comparison")
model_names, model_scores = zip(*model_performance)

plt.figure(figsize=(12, 6))
plt.barh(model_names, model_scores, color='skyblue')
plt.xlabel("R² Score")
plt.ylabel("Model")
plt.title("Performance of Different Models")
plt.xlim(-1, 1)  # Displaying both positive and negative R² values
st.pyplot(plt)

# -------------------- SECTION 2: USER DATA HEATMAP --------------------

st.header("User Data Visualizations")

# Allow the user to upload a CSV file for predictions
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    st.write("User Data:")
    st.write(user_data)

    # If 'Traffic_Volume' exists in the user dataset, drop it (it's the target column)
    if 'Traffic_Volume' in user_data.columns:
        user_data = user_data.drop(columns=['Traffic_Volume'])

    # Preprocess user data (same as before)
    user_data['Date'] = pd.to_datetime(user_data['Date'], errors='coerce')
    user_data['Year'] = user_data['Date'].dt.year
    user_data['Month'] = user_data['Date'].dt.month
    user_data['Day'] = user_data['Date'].dt.day
    user_data['Weekday'] = user_data['Date'].dt.weekday

    # Encode categorical columns (same encoding used during training)
    for col in categorical_columns:
        if col in user_data.columns:
            user_data[col] = user_data[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])  
            user_data[col] = encoders[col].transform(user_data[col])

    # Fill missing values in non-target columns
    non_target_columns = [col for col in user_data.columns if col != 'Traffic_Volume']
    user_data = user_data.fillna(user_data.mean())

    # Apply preprocessing (scaling) to user data
    user_data_preprocessed = preprocessor.transform(user_data)

    # Predict traffic volume using the best model
    user_predictions = best_model.predict(user_data_preprocessed)

    # Display predictions
    st.subheader("Predicted Traffic Volumes")
    prediction_df = pd.DataFrame(user_data, columns=non_target_columns)
    prediction_df['Predicted Traffic Volume'] = user_predictions
    st.write(prediction_df)

    # Display detailed model performance in a notepad-like box
    st.subheader("Model Performance Summary")
    with st.expander("Click to view performance summary"):
        st.write(f"Best Model: {best_model_name}")
        st.write(f"Test R²: {test_r2}")
        st.write(f"Test RMSE: {rmse}")
        st.write(f"Test Adjusted R²: {adj_r2}")

        # Show model comparison details
        st.write(f"R² Scores for all models: {model_performance}")

