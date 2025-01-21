import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# Paths for saving models and encoders
MODEL_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\traffic_regression_model.pkl"
ENCODER_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\label_encoders.pkl"

# Load the dataset
data = pd.read_csv("C:/Advanced projects/Bangalore_Traffic/Banglore_traffic_Dataset.csv")

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.weekday
data = data.drop(columns=['Date'])

# Handle categorical columns by encoding them
categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Traffic Signal Compliance', 'Roadwork and Construction Activity']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].fillna('missing'))  # Fill missing values before encoding
    encoders[col] = le

# Save the encoders
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoders, f)

# Fill numeric missing values
data = data.fillna(data.mean())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Splitting the data into train/test
X = data.drop(columns=['Traffic_Volume'])
y = data['Traffic_Volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Feature scaling for numeric columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
numeric_transformer = StandardScaler()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', numeric_transformer, numeric_features),
        ('cat', 'passthrough', categorical_columns)  # No transformation for categorical columns, as they're already encoded
    ])

# Define the models
models = {
    'Random Forest': RandomForestRegressor(random_state=123),
    'Linear Regression': LinearRegression(),
    'Support Vector Regressor': SVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=123),
    'K-Nearest Neighbors': KNeighborsRegressor()
}

# Model performance tracking
best_model = None
best_score = -float('inf')
best_model_name = None

# Evaluate all models
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)

    # Predict and evaluate performance
    y_pred = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)

    if test_r2 > best_score:
        best_score = test_r2
        best_model = pipeline
        best_model_name = name

# Save the best model to the pickle file
with open(MODEL_PATH, 'wb') as model_file:
    pickle.dump(best_model, model_file)

# Evaluate the best model on the test set
y_pred_best = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_best)
test_r2 = r2_score(y_test, y_pred_best)

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
st.write(f"Mean Squared Error (MSE) on Training Data: {train_mse}")
st.write(f"R² Score on Training Data: {train_r2}")

# -------------------- SECTION 2: USER DATA HEATMAP --------------------

st.header("User Data Visualizations")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    st.write("User Data:")
    st.write(user_data)
    
    # Check if 'Traffic_Volume' exists in the user dataset and drop it if it's empty
    if 'Traffic_Volume' in user_data.columns:
        user_data = user_data.drop(columns=['Traffic_Volume'])

    # Preprocess user data (same as before)
    user_data['Date'] = pd.to_datetime(user_data['Date'], errors='coerce')
    user_data['Year'] = user_data['Date'].dt.year
    user_data['Month'] = user_data['Date'].dt.month
    user_data['Day'] = user_data['Date'].dt.day
    user_data['Weekday'] = user_data['Date'].dt.weekday
    user_data = user_data.drop(columns=['Date'])

    # Encode categorical columns (same encoding used during training)
    for col in categorical_columns:
        if col in user_data.columns:
            user_data[col] = user_data[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])  # Handle unseen labels
            user_data[col] = encoders[col].transform(user_data[col])

    # Fill missing values in non-target columns
    non_target_columns = [col for col in user_data.columns if col != 'Traffic_Volume']
    user_data[non_target_columns] = user_data[non_target_columns].fillna(user_data[non_target_columns].mean())
    for col in categorical_columns:
        if col in user_data.columns:
            user_data[col] = user_data[col].fillna(user_data[col].mode()[0])

    # Display heatmap for user data
    st.subheader("User Data Heatmap")
    user_corr_matrix = user_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(user_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    st.pyplot(plt)

    # -------------------- SECTION 3: MODEL PERFORMANCE ON USER DATA --------------------

    st.subheader("Model Performance on User Data")
    try:
        user_predictions = best_model.predict(user_data)
        user_data['Predicted_Traffic_Volume'] = user_predictions

        st.write("Predictions for User Data:")
        st.write(user_data[['Predicted_Traffic_Volume']])
    except Exception as e:
        st.error(f"An error occurred while making predictions: {e}")

    # Option to download the predictions
    csv = user_data.to_csv(index=False)
    st.download_button("Download Predictions", data=csv, file_name="user_predictions.csv", mime="text/csv")

#  Date column is dropped and is not working if present , pls go over to 10th iteration to test new code with date column intact 