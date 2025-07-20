This tool predicts the risk of heart disease using key health metrics. It analyzes user-provided data or an uploaded Excel file to generate risk assessments with visualizations.

Features
-Predicts heart disease risk using logistic regression
-Accepts Excel uploads with required health metrics
-Interactive risk gauge and visualizations
-Personalized health recommendations

How to Use
Install requirements:
pip install -r requirements.txt  
Run the app:
streamlit run heart_disease_predictor.py 


Input data:
Manually enter metrics or
Upload an Excel file with columns:
Age
RestingBP (mmHg)
Cholesterol (mg/dL)
FastingBS>120 (Yes/No)
ExerciseAngina (Yes/No)
HeartDisease (0/1 for training data)

Note:
Invalid/missing data rows are automatically removed.
Predictions are estimates, not medical diagnoses.
