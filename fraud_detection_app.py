import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

def load_and_preprocess_data(data):
    """Load and preprocess the credit card transaction data"""
    # Assuming data has these columns: 
    # Amount, Time, V1-V28 (anonymous features), Class (1 for fraud, 0 for normal)
    
    # Scale the Amount and Time features
    scaler = StandardScaler()
    data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['Time_scaled'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    
    # Drop original Amount and Time columns
    data = data.drop(['Time', 'Amount'], axis=1)
    
    return data

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return their performance metrics"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'model': model
        }
    
    return results

def main():
    st.title("Credit Card Fraud Detection System")
    
    # File upload
    uploaded_file = st.file_uploader("creditcard.csv", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)
        
        # Display sample of raw data
        st.subheader("Sample of Raw Data")
        st.write(data.head())
        
        # Preprocess data
        processed_data = load_and_preprocess_data(data)
        
        # Split features and target
        X = processed_data.drop('Class', axis=1)
        y = processed_data['Class']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models and get results
        results = train_models(X_train, X_test, y_train, y_test)
        
        # Display results
        st.header("Model Performance Comparison")
        
        # Create columns for different metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Display accuracy comparison
            st.subheader("Model Accuracies")
            accuracies = {name: res['accuracy'] for name, res in results.items()}
            acc_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy'])
            st.bar_chart(acc_df)
        
        with col2:
            # Display confusion matrices
            st.subheader("Confusion Matrices")
            selected_model = st.selectbox(
                "Select model to view confusion matrix",
                list(results.keys())
            )
            
            # Plot confusion matrix
            fig, ax = plt.subplots()
            sns.heatmap(
                results[selected_model]['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax
            )
            ax.set_title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
        
        # Display detailed classification report
        st.subheader("Detailed Classification Report")
        selected_model_report = st.selectbox(
            "Select model to view classification report",
            list(results.keys()),
            key="report_selector"
        )
        st.text(results[selected_model_report]['classification_report'])
        
        # Add real-time prediction capability
        st.header("Real-time Prediction")
        st.write("Enter transaction details for prediction")
        
        # Create input fields for features
        input_data = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_data['Amount'] = st.number_input("Transaction Amount", value=0.0)
        with col2:
            input_data['Time'] = st.number_input("Time (seconds from first transaction)", value=0)
            
        # Add anonymous features V1-V28
        st.write("Anonymous Features (V1-V28)")
        cols = st.columns(4)
        for i in range(28):
            with cols[i % 4]:
                input_data[f'V{i+1}'] = st.number_input(f'V{i+1}', value=0.0)
        
        if st.button("Predict"):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_df_scaled = load_and_preprocess_data(input_df)
            
            # Make predictions with all models
            st.subheader("Prediction Results")
            for name, result in results.items():
                prediction = result['model'].predict(input_df_scaled)
                probability = result['model'].predict_proba(input_df_scaled)
                
                st.write(f"{name}:")
                st.write(f"Prediction: {'Fraudulent' if prediction[0] == 1 else 'Legitimate'}")
                st.write(f"Probability of fraud: {probability[0][1]:.4f}")
                st.write("---")

if __name__ == "__main__":
    main()