# Heart Disease Risk Predictor - Complete Implementation
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns, na_values='?')
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    df = df.dropna()
    return df

def preprocess_inputs(df, age, sex, trestbps, chol, fbs, exang, thalach=150, oldpeak=1.0):
    # Convert inputs to model format
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0
    
    # Create input array with median values for unused features
    input_data = {
        'age': age,
        'sex': sex,
        'cp': df['cp'].median(),  # Using median for missing features
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': df['restecg'].median(),
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': df['slope'].median(),
        'ca': df['ca'].median(),
        'thal': df['thal'].median()
    }
    
    return pd.DataFrame([input_data])

# Visualization functions
def plot_feature_importance(model, features):
    importance = model.coef_[0] if hasattr(model, 'coef_') else model.feature_importances_
    fig = px.bar(x=features, y=importance, 
                 labels={'x':'Features', 'y':'Importance'},
                 title='Feature Importance for Heart Disease Prediction')
    fig.update_layout(template='plotly_dark')
    return fig

def plot_radar_chart(patient_data, healthy_ranges):
    categories = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST Depression']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=patient_data,
        theta=categories,
        fill='toself',
        name='Your Metrics'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=healthy_ranges,
        theta=categories,
        fill='toself',
        name='Healthy Ranges'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Your Health Metrics vs. Healthy Ranges"
    )
    
    return fig

def plot_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability*100,
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
                'value': probability*100}}))
    
    return fig

def plot_age_risk_distribution(df):
    fig = px.scatter(df, x='age', y='target', color='sex',
                     trendline="lowess",
                     title="Age vs. Heart Disease Risk",
                     labels={'age': 'Age', 'target': 'Disease Presence', 'sex': 'Gender'},
                     color_discrete_map={0: 'pink', 1: 'blue'})
    fig.update_traces(marker=dict(size=10, opacity=0.6))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    return fig

def show_recommendations(probability, age, chol, bp):
    st.subheader("Personalized Recommendations")
    
    if probability < 0.3:
        st.success("✅ Your heart disease risk is low. Maintain your healthy habits!")
    elif probability < 0.7:
        st.warning("⚠️ Your heart disease risk is moderate. Consider these improvements:")
    else:
        st.error("❗ Your heart disease risk is high. Please consult a doctor and consider these changes:")
    
    if probability >= 0.3:
        if bp > 130:
            st.write("- Work on lowering your blood pressure through diet and exercise")
        if chol > 200:
            st.write("- Reduce cholesterol through dietary changes (less saturated fat)")
        if age > 45:
            st.write("- Regular cardiovascular exercise is especially important as you age")
        st.write("- Quit smoking if you currently smoke")
        st.write("- Manage stress through meditation or relaxation techniques")

# Main application
def main():
    st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")
    
    # Load data and train model
    df = load_data()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    # App title and description
    st.title("❤️ Heart Disease Risk Predictor")
    st.write("""
    This tool assesses your risk of heart disease based on key health metrics. 
    Enter your information below to get a personalized risk assessment.
    """)
    
    # Layout columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.form("user_input"):
            st.header("Your Health Metrics")
            age = st.slider("Age", 20, 80, 45)
            sex = st.selectbox("Gender", ["Male", "Female"])
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
            oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0)
            
            submitted = st.form_submit_button("Calculate Risk")
    
    if submitted:
        # Preprocess inputs and make prediction
        input_df = preprocess_inputs(df, age, sex, trestbps, chol, fbs, exang, thalach, oldpeak)
        input_scaled = scaler.transform(input_df)
        proba = lr.predict_proba(input_scaled)[0][1]
        
        with col2:
            st.plotly_chart(plot_risk_gauge(proba), use_container_width=True)
            
            # Show recommendations
            show_recommendations(proba, age, chol, trestbps)
            
            # Show feature importance
            st.plotly_chart(plot_feature_importance(lr, X.columns), use_container_width=True)
        
        # Additional visualizations
        st.subheader("Health Insights")
        col3, col4 = st.columns(2)
        
        with col3:
            # Radar chart with sample data
            patient_metrics = [age, trestbps, chol, thalach, oldpeak]
            healthy_ranges = [40, 120, 200, 175, 0.5]  # Example healthy ranges
            st.plotly_chart(plot_radar_chart(patient_metrics, healthy_ranges), use_container_width=True)
        
        with col4:
            st.plotly_chart(plot_age_risk_distribution(df), use_container_width=True)
    
    # Model info section
    with st.expander("About This Tool"):
        st.write("""
        **Model Information:**
        - Algorithm: Logistic Regression
        - Accuracy: {:.2f}% (on test data)
        - Dataset: Cleveland Heart Disease Dataset from UCI Machine Learning Repository
        
        **Healthy Ranges:**
        - Blood Pressure: <120/80 mm Hg
        - Cholesterol: <200 mg/dL
        - Fasting Blood Sugar: <100 mg/dL
        
        **Note:** This tool provides risk estimates, not medical diagnoses. 
        Always consult a healthcare professional for medical advice.
        """.format(accuracy_score(y_test, lr.predict(X_test_scaled)) * 100))

if __name__ == "__main__":
    main()
