import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

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

# Fill missing values
data = data.fillna(data.mean())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Train-test split
X = data.drop(columns=['Traffic_Volume'])
y = data['Traffic_Volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR()
}

# Initialize dictionaries to store performance results
model_results = {}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_results[model_name] = {"MSE": mse, "R²": r2}
    
# Convert the results to a DataFrame for easy visualization
results_df = pd.DataFrame(model_results).T
results_df = results_df.sort_values(by="R²", ascending=False)

# Display the results in Streamlit
st.write("Model Performance Comparison")
st.write(results_df)

# Plot a barplot for model comparison
sns.barplot(x=results_df.index, y=results_df['R²'], palette='viridis')
plt.title('Model R² Score Comparison')
plt.xlabel('Model')
plt.ylabel('R² Score')
st.pyplot()

# Save the best model based on R² score
best_model_name = results_df.index[0]
best_model = models[best_model_name]
with open("best_model.pkl", 'wb') as f:
    pickle.dump(best_model, f)
st.write(f"Best model is {best_model_name} with R²: {results_df.iloc[0]['R²']}")

# Streamlit interface for prediction
st.title("Bangalore Traffic Prediction App")

# File uploader for input data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Input Data:")
    st.write(input_data)

    try:
        # Preprocess the input data
        input_data['Date'] = pd.to_datetime(input_data['Date'], errors='coerce')
        input_data['Year'] = input_data['Date'].dt.year
        input_data['Month'] = input_data['Date'].dt.month
        input_data['Day'] = input_data['Date'].dt.day
        input_data['Weekday'] = input_data['Date'].dt.weekday
        input_data = input_data.drop(columns=['Date'])

        # Encode categorical columns
        for col in categorical_columns:
            if col in input_data.columns:
                input_data[col] = input_data[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
                input_data[col] = encoders[col].transform(input_data[col])

        # Fill missing values
        non_target_columns = [col for col in input_data.columns if col != 'Traffic_Volume']
        input_data[non_target_columns] = input_data[non_target_columns].fillna(input_data[non_target_columns].mean())

        # Predict using the best model
        predictions = best_model.predict(input_data.drop(columns=['Traffic_Volume'], errors='ignore'))
        input_data['Predicted Traffic Volume'] = predictions

        st.write("Predictions:")
        st.write(input_data)

        # Option to download the predictions
        csv = input_data.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="traffic_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"An error occurred: {e}")
