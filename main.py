
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

REQUIRED_COLUMNS = [
    'age', 
    'restingbp', 
    'cholesterol', 
    'fastingbs>120', 
    'exerciseangina', 
    'heartdisease'
]

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.lower()
            
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None
                
            return clean_data(df)
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    else:
        st.info("Using sample data since no file was uploaded")
        data = {
            'age': [52, 38, 45, 60],
            'restingbp': [125, 138, 110, 140],
            'cholesterol': [240, 182, 210, 318],
            'fastingbs>120': ["No", "Yes", "No", "Yes"],
            'exerciseangina': ["No", "Yes", "No", "Yes"],
            'heartdisease': [0, 1, 0, 1]
        }
        return pd.DataFrame(data)

def clean_data(df):
    """Clean and standardize the dataset with NA/NaN handling"""
    numeric_cols = ['age', 'restingbp', 'cholesterol']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    if 'fastingbs>120' in df.columns:
        df['fastingbs>120'] = df['fastingbs>120'].apply(
            lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else (
                0 if str(x).lower() in ['no', '0', 'false'] else np.nan
            )
        )
    
    if 'exerciseangina' in df.columns:
        df['exerciseangina'] = df['exerciseangina'].apply(
            lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else (
                0 if str(x).lower() in ['no', '0', 'false'] else np.nan
            )
        )
    
    if 'heartdisease' in df.columns:
        df['heartdisease'] = pd.to_numeric(df['heartdisease'], errors='coerce')
        df['heartdisease'] = df['heartdisease'].replace([np.inf, -np.inf], np.nan)
        df['heartdisease'] = df['heartdisease'].clip(0, 1)
    
    df_cleaned = df.dropna()
    
    if len(df_cleaned) < len(df):
        st.warning(f"Removed {len(df) - len(df_cleaned)} rows with missing/invalid values")
    
    return df_cleaned

def preprocess_inputs(age, restingbp, cholesterol, fastingbs, exerciseangina):
    """Prepare user inputs for prediction"""
    return pd.DataFrame([{
        'age': age,
        'restingbp': restingbp,
        'cholesterol': cholesterol,
        'fastingbs>120': 1 if fastingbs == "Yes" else 0,
        'exerciseangina': 1 if exerciseangina == "Yes" else 0
    }])

def plot_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    return fig

def show_recommendations(probability, age, cholesterol):
    st.subheader("Recommendations")
    if probability < 0.3:
        st.success("Low risk detected. Maintain healthy habits!")
    elif probability < 0.7:
        st.warning("Moderate risk detected. Consider:")
        st.write("- Cholesterol management (current: {} mg/dL)".format(cholesterol))
        if age > 45:
            st.write("- Regular cardiovascular checkups")
    else:
        st.error("High risk detected. Please consult a doctor immediately.")
        st.write("- Urgent cholesterol management needed (current: {} mg/dL)".format(cholesterol))
        st.write("- Comprehensive cardiac evaluation recommended")

def main():
    st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
    st.title("Heart Disease Risk Predictor")
    
    # File upload
    with st.expander("Upload Data (Excel with required columns)"):
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
        st.write("""
        Required columns:
        - Age (years)
        - RestingBP (mm Hg)
        - Cholesterol (mg/dL)
        - FastingBS>120 (Yes/No)
        - ExerciseAngina (Yes/No)
        - HeartDisease (0/1)
        """)
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        # Prepare data
        X = df[['age', 'restingbp', 'cholesterol', 'fastingbs>120', 'exerciseangina']]
        y = df['heartdisease']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # User input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age (years)", 20, 100, 45)
                restingbp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
                
            with col2:
                cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
                fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
                exerciseangina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            
            submitted = st.form_submit_button("Calculate Risk")
        
        if submitted:
            # Make prediction
            input_df = preprocess_inputs(age, restingbp, cholesterol, fastingbs, exerciseangina)
            input_scaled = scaler.transform(input_df)
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Show results
            st.plotly_chart(plot_risk_gauge(probability), use_container_width=True)
            show_recommendations(probability, age, cholesterol)
            
            # Show model info
            with st.expander("Model Information"):
                st.write(f"Accuracy: {accuracy_score(y_test, model.predict(X_test_scaled)):.2%}")
                st.write("Features used:")
                st.write("- Age")
                st.write("- Resting Blood Pressure")
                st.write("- Cholesterol")
                st.write("- Fasting Blood Sugar >120 mg/dL")
                st.write("- Exercise Induced Angina")

if __name__ == "__main__":
    main()
