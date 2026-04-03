import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = {
    'Age': [25, 30, 35, 40, 28, 50],
    'MonthlyIncome': [3000, 4000, 5000, 6000, 3500, 8000],
    'JobSatisfaction': [1, 2, 3, 4, 2, 3],
    'YearsAtCompany': [1, 3, 5, 10, 2, 15],
    'OverTime': [0, 1, 0, 1, 0, 1],
    'Attrition': [1, 0, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop('Attrition', axis=1)
y = df['Attrition']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
print("Model Ready ✅")