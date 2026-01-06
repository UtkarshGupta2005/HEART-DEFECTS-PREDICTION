Problem Statement:
Heart defects are among the leading causes of mortality worldwide. Early detection using machine learning can assist healthcare professionals in identifying high-risk patients and improving diagnostic accuracy.

Objective:
To build a predictive model that classifies whether a patient is likely to have a heart defect based on clinical and physiological parameters.

Dataset:
Patient medical records
Features include:
Age
Sex
Blood pressure(Systolic & Diastolic)
Cholesterol level
Glucose level
Fasting blood sugar
Maximum heart rate
Prevalent Stroke & Hypertension

Methodology:
Data Preprocessing
Handling missing values using Iterative Imputer
Feature scaling using Robust Scaler
Encoding categorical variables
Exploratory Data Analysis (EDA)
Feature importance visualization

Model Training:
Random Forest

Hyperparameter Tuning:
GridSearchCV
RandomSearchCV
Bayesian Optimization

Model Evaluation:
Accuracy
Precision, Recall, F1-score
Confusion Matrix

Results:
Random Forest achieved 84.98% accuracy
Bayesian Optimization improved model accuracy and generalization

Tech Stack:
Python
NumPy, Pandas
Scikit-learn
Matplotlib, Seaborn

Use Cases:
Clinical decision support systems
Preventive healthcare analytics
Medical research assistance
