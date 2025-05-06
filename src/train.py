import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Cargar dataset simple (simulamos si no existe)
path = "data/churn.csv"
if not os.path.exists(path):
    df = pd.DataFrame({
        "age": [25, 40, 35, 50],
        "balance": [1000, 2000, 1500, 3000],
        "churn": [0, 1, 0, 1]
    })
    df.to_csv(path, index=False)
else:
    df = pd.read_csv(path)

# Preprocesamiento simple
X = df.drop("churn", axis=1)
y = df["churn"]

# Entrenar modelo
model = RandomForestClassifier()
model.fit(X, y)

# Guardar modelo
joblib.dump(model, "model/model.pkl")
