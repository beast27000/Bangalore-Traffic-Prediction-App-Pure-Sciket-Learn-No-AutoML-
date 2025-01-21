import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
import pickle
import streamlit as st

# Step 1: Load the dataset
dataset = pd.read_csv(r'C:\Advanced projects\Pure Sckiet learn projects\Bangalore_traffic\Banglore_traffic_Dataset.csv')

# Step 2: Preprocess the 'Date' column
def preprocess_date_column(data):
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Handle invalid dates
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
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns)  # handle unknown categories
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

# Save pipelines to pickle files
for model_name, pipeline in pipelines.items():
    with open(f'{model_name}_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

# Convert results to a DataFrame for easier visualization
results_df = pd.DataFrame(results).sort_values(by='R²', ascending=False)

# Step 8: Visualize the results
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='R²', y='Model', hue='Model', palette='viridis', legend=False)
plt.title('Model Performance (R² Score)')
plt.xlabel('R² Score')
plt.ylabel('Model')
# Use Streamlit to display the plot instead of plt.show()
st.pyplot(plt)

# Step 9: Correlation heatmap (Non-preprocessed data)
numerical_columns = dataset.select_dtypes(include=["number"]).columns  # Select only numerical columns
corr_matrix = dataset[numerical_columns].corr()  # Compute correlation matrix

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, ax=ax)
plt.title("Correlation Heatmap (Non-Preprocessed Data)")
# Use Streamlit to display the plot instead of plt.show()
st.pyplot(fig)

# Step 10: Streamlit app
def run_streamlit_app():
    st.title("Bangalore Traffic Volume Prediction")

    # Upload user data
    uploaded_file = st.file_uploader("Upload a CSV file with traffic data:", type="csv")

    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        original_data = user_data.copy()  # Keep a copy of original data

        # Preprocess user data (same as training data preprocessing)
        user_data = preprocess_date_column(user_data)  # Ensure date column is processed
        user_data = user_data.drop(columns=[target_column], errors='ignore')  # Drop target column if present

        # Load the best pipeline based on the model performance (highest R²)
        best_model = results_df.iloc[0]['Model']
        with open(f'{best_model}_pipeline.pkl', 'rb') as f:
            best_pipeline = pickle.load(f)

        # Predict using the best pipeline
        predictions = best_pipeline.predict(user_data)

        # Add predictions to the original data
        original_data[target_column] = predictions
        st.write("Predictions:", original_data)

        # Visualization options
        st.subheader("Visualizations")
        if st.button("Show Correlation Heatmap"):
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, ax=ax)
            st.pyplot(fig)

        if st.button("Show Model Performance"):
            st.write("### Model Performance")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=results_df, x='R²', y='Model', palette='viridis', ax=ax)
            ax.set_title('Model Performance (R² Score)')
            ax.set_xlabel('R² Score')
            ax.set_ylabel('Model')
            st.pyplot(fig)

# Run the Streamlit app
if __name__ == "__main__":
    run_streamlit_app()
