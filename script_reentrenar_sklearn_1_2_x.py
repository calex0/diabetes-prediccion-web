import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ============================================================
# 1. Cargar dataset preparado (train_prepared.csv)
# ============================================================

df = pd.read_csv("train_prepared.csv")

columnas_rl = [
    "_RFHLTH","_AGE_G","_BMI5CAT","_RACEG21","WTCHSALT",
    "ASBIRDUC","CTELNUM1","CTELENUM","SCNTMEL1"
]

X = df[columnas_rl]
y = df["diabetes"]

# ============================================================
# 2. Preprocesamiento compatible con scikit-learn 1.2.x
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), columnas_rl)
    ]
)

# ============================================================
# 3. Modelo RL compatible con 1.2.x
# ============================================================

clf = LogisticRegression(
    class_weight="balanced",
    solver="liblinear",
    max_iter=2000,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preproc', preprocessor),
    ('model', clf)
])

# ============================================================
# 4. Entrenar
# ============================================================

pipeline.fit(X, y)

# ============================================================
# 5. Guardar pipeline y columnas
# ============================================================

joblib.dump(pipeline, "modelo_rl.pkl")
joblib.dump(columnas_rl, "columnas_rl.pkl")

# ============================================================
# 6. Generar X_background para SHAP
# ============================================================

# Usamos 200 muestras aleatorias del dataset transformado
X_background = preprocessor.transform(X.sample(200, random_state=42))
joblib.dump(X_background, "X_background.pkl")

print("Entrenamiento completado con scikit-learn 1.2.x")
