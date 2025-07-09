import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

# =================== 1. Create Dataset ===================
data = {
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Married': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Graduate',
                  'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate'],
    'ApplicantIncome': [5000, 3000, 4000, 6000, 2500, 4200, 3500, 8000, 2700, 6500],
    'LoanAmount': [130, 100, 120, 150, 110, 115, 100, 160, 90, 155],
    'Credit_History': [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    'Loan_Status': ['Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y']
}
df = pd.DataFrame(data)

# =================== 2. Preprocessing ===================
# Handle missing values (example: fill with mode or median)
df.fillna(method='ffill', inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# =================== 3. Balance Dataset ===================
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

# =================== 4. Train-Test Split ===================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# =================== 5. Scaling ===================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# =================== 6. Train Model (Logistic Regression) ===================
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)

# =================== 7. Evaluate Model ===================
y_pred = logistic_reg.predict(X_val)
print("\n=== Logistic Regression Evaluation ===")
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

try:
    roc = roc_auc_score(y_val, y_pred)
    print("ROC AUC Score:", roc)
except Exception as e:
    print("ROC AUC Score Error:", e)

# =================== 8. User Prediction ===================
print("\n=== Predict Loan Approval ===")
try:
    # Input from user
    gender = input("Enter Gender (Male/Female): ").strip().lower()
    married = input("Married? (Yes/No): ").strip().lower()
    education = input("Education (Graduate/Not Graduate): ").strip().lower()
    income = float(input("Applicant Income: "))
    loan_amt = float(input("Loan Amount: "))
    credit_hist = int(input("Credit History (1=Good, 0=Bad): "))

    # Encode input
    gender_encoded = 1 if gender == 'male' else 0
    married_encoded = 1 if married == 'yes' else 0
    education_encoded = 1 if education == 'graduate' else 0

    # Create input array
    user_input = np.array([[gender_encoded, married_encoded, education_encoded,
                            income, loan_amt, credit_hist]])
    user_input_scaled = scaler.transform(user_input)

    # Predict
    pred = logistic_reg.predict(user_input_scaled)[0]

    if pred == 1:
        print("✅ Prediction: Loan Approved ✅")
    else:
        print("❌ Prediction: Loan Not Approved ❌")

except Exception as e:
    print("Error in prediction:", e)
