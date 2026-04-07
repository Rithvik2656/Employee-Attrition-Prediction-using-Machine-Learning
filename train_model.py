import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Kaggle dataset

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Convert target column (Yes/No → 1/0)

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Convert OverTime column

df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

# Select important features

features = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany', 'OverTime']

X = df[features]
y = df['Attrition']

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model

joblib.dump(model, "model.pkl")

print("Model trained and saved successfully ✅")
