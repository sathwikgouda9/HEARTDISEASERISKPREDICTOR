Heart Disease Risk Predictor

This is a Python application that predicts the risk of heart disease based on key health metrics. The tool uses machine learning to provide personalized risk assessments along with visual explanations of the factors contributing to the prediction.

Project Overview:
- Loads and processes heart disease data from the UCI Machine Learning Repository
- Trains a logistic regression model to predict heart disease risk
- Provides an interactive web interface using Streamlit
- Generates visualizations to help users understand their risk factors

Key Features:
- Input form for health metrics including age, blood pressure, and cholesterol
- Real-time risk calculation displayed as a percentage
- Visual gauge showing risk level (low, medium, high)
- Radar chart comparing user metrics to healthy ranges
- Personalized health recommendations based on results

How to Use:
1. Install the required packages (see requirements.txt)
2. Run the application using: streamlit run heart_disease_predictor.py
3. Enter your health information in the sidebar
4. View your risk assessment and recommendations

Data Source:
The application uses the processed Cleveland Heart Disease dataset from the UCI Machine Learning Repository. This dataset contains medical information from 303 patients along with heart disease diagnosis.

Limitations:
- This tool provides risk estimates only, not medical diagnoses
- Accuracy depends on the quality of input data
- Should not be used as a substitute for professional medical advice

For questions or issues, please open an issue in the project repository.
