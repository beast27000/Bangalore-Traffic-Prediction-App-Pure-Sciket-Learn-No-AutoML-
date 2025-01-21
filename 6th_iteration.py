import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Save the encoders
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoders, f)

# Fill missing values
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

# Model pipeline: Example using RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=123))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model to a file
with open(MODEL_PATH, 'wb') as model_file:
    pickle.dump(pipeline, model_file)

# Evaluate the model on the test set
y_pred = pipeline.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

# Streamlit App for Predictions and Performance Metrics
st.title("Bangalore Traffic Prediction App")

# -------------------- SECTION 1: MODEL PERFORMANCE --------------------

st.header("Model Performance Metrics")

if st.button("Show Model Performance"):
    st.write(f"Mean Squared Error (MSE): {test_mse}")
    st.write(f"R² Score: {test_r2}")
    
    # Show performance as a bar chart
    performance_data = {
        'Metric': ['MSE', 'R²'],
        'Value': [test_mse, test_r2]
    }
    performance_df = pd.DataFrame(performance_data)
    
    # Plot the bar chart
    st.bar_chart(performance_df.set_index('Metric'))

# -------------------- SECTION 2: HEATMAP --------------------

st.header("Data Heatmap")

if st.button("Generate Data Heatmap"):
    # Data Profiling - Generating a heatmap of the correlation matrix
    st.write("Generating Heatmap...")
    corr_matrix = data.corr()  # Compute correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    st.pyplot(plt)

# -------------------- SECTION 3: PREDICTIONS --------------------

st.header("Make Predictions")

# File uploader for input data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Input Data:")
    st.write(input_data)

    try:
        # Preprocess input data (same as before)
        input_data['Date'] = pd.to_datetime(input_data['Date'], errors='coerce')
        input_data['Year'] = input_data['Date'].dt.year
        input_data['Month'] = input_data['Date'].dt.month
        input_data['Day'] = input_data['Date'].dt.day
        input_data['Weekday'] = input_data['Date'].dt.weekday
        input_data = input_data.drop(columns=['Date'])

        # Encode categorical columns (same encoding used during training)
        for col in categorical_columns:
            if col in input_data.columns:
                input_data[col] = input_data[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])  # Handle unseen labels
                input_data[col] = encoders[col].transform(input_data[col])

        # Fill missing values in non-target columns
        non_target_columns = [col for col in input_data.columns if col != 'Traffic_Volume']
        input_data[non_target_columns] = input_data[non_target_columns].fillna(input_data[non_target_columns].mean())
        for col in categorical_columns:
            if col in input_data.columns:
                input_data[col] = input_data[col].fillna(input_data[col].mode()[0])

        # Load the saved model for prediction
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)

        # Make predictions
        predictions = model.predict(input_data.drop(columns=['Traffic_Volume']))
        input_data['Traffic_Volume'] = predictions  # Add predictions to the DataFrame

        st.write("Predictions:")
        st.write(input_data)

        # Option to download the predictions
        csv = input_data.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="traffic_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
