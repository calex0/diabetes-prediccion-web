# app_diabetes.py
# Cribado de diabetes tipo 2 (diabetes sí/no) con RL + SHAP y Tkinter

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

# ============================================================
# 1. Mapeo español → columnas técnicas
# ============================================================

mapa_es = {
    "_RFHLTH": "Salud física/mental general",
    "_AGE_G": "Grupo de edad",
    "_BMI5CAT": "Categoría de IMC",
    "_RACEG21": "Grupo racial/étnico",
    "WTCHSALT": "Control de sal",
    "ASBIRDUC": "Reducción del consumo de alcohol",
    "CTELNUM1": "Número de teléfonos",
    "CTELENUM": "Disponibilidad de teléfono",
    "SCNTMEL1": "Días de mala salud mental"
}

# Cargar columnas reales del modelo
columnas_rl = joblib.load("columnas_rl.pkl")

# Crear FEATURE_NAMES en español respetando el orden real
FEATURE_NAMES = [mapa_es[col] for col in columnas_rl]


# ============================================================
# 2. Combobox con categorías BRFSS (siguen usando claves técnicas)
# ============================================================

opciones_combobox = {
    "_RFHLTH": {
        0: "Good/Excellent",
        1: "Fair",
        2: "Poor"
    },

    "_AGE_G": {
        0: "18-24",
        1: "25-34",
        2: "35-44",
        3: "45-54",
        4: "55-64",
        5: "65+"
    },

    "_BMI5CAT": {
        0: "Underweight",
        1: "Normal",
        2: "Overweight",
        3: "Obese"
    },

    "_RACEG21": {
        0: "White",
        1: "Black",
        2: "Other"
    },

    "WTCHSALT": {
        0: "No",
        1: "Yes",
        2: "Don't know",
        3: "Refused"
    },

    "ASBIRDUC": {
        0: "No",
        1: "Yes",
        2: "Don't know",
        3: "Refused"
    },

    "CTELNUM1": {
        0: "None",
        1: "One phone",
        2: "Two or more"
    },

    "CTELENUM": {
        0: "No",
        1: "Yes",
        2: "Don't know"
    },

    "SCNTMEL1": {
        0: "0 days",
        1: "1–7 days",
        2: "8–14 days",
        3: "15–21 days",
        4: "22–30 days",
        5: "31–60 days",
        6: "61–90 days",
        7: "90+ days"
    }
}

# ============================================================
# 3. Cargar pipeline RL (preproc + model) y background para SHAP
# ============================================================

pipeline = joblib.load("modelo_rl.pkl")
Xb_trans = joblib.load("X_background.pkl")

preproc = pipeline.named_steps["preproc"]
clf = pipeline.named_steps["model"]

columnas_transformadas = preproc.get_feature_names_out()

explainer = shap.LinearExplainer(
    clf,
    Xb_trans,
    feature_perturbation="interventional"
)

# ============================================================
# 4. Funciones de predicción y explicación
# ============================================================

def construir_vector_entrada(valores_dict):
    data = {}
    for col in columnas_rl:
        if col in valores_dict:
            data[col] = [float(valores_dict[col])]
        else:
            data[col] = [0.0]
    return pd.DataFrame(data)


def agrupar_shap_por_variable_original(shap_values_full):
    shap_por_var = {}
    for col in columnas_rl:
        cols_var = [i for i, name in enumerate(columnas_transformadas)
                    if col in name]
        if not cols_var:
            shap_por_var[col] = 0.0
        else:
            shap_por_var[col] = shap_values_full[cols_var].sum()
    return np.array([shap_por_var[col] for col in columnas_rl])


def predecir_y_explicar(valores_dict):
    x = construir_vector_entrada(valores_dict)

    prob = pipeline.predict_proba(x)[0, 1]
    clase = int(pipeline.predict(x)[0])

    x_trans = preproc.transform(x)
    shap_values_full = explainer.shap_values(x_trans)[0]

    shap_values_rfe = agrupar_shap_por_variable_original(shap_values_full)

    return prob, clase, shap_values_rfe, FEATURE_NAMES


def mostrar_grafico_shap(shap_values, feature_names):
    plt.figure(figsize=(7, 4))
    indices = np.argsort(np.abs(shap_values))[::-1]
    ordered_features = [feature_names[i] for i in indices]
    ordered_shap = shap_values[indices]
    colors = ["red" if v > 0 else "blue" for v in ordered_shap]

    plt.barh(ordered_features[::-1], ordered_shap[::-1], color=colors[::-1])
    plt.xlabel("Valor SHAP (contribución al riesgo de diabetes)")
    plt.title("Explicación SHAP de la predicción")
    plt.tight_layout()
    plt.show()


def generar_texto_explicativo(prob, clase, shap_values, feature_names):
    if prob < 0.20:
        riesgo_cat = "bajo"
    elif prob < 0.40:
        riesgo_cat = "moderado"
    else:
        riesgo_cat = "alto"

    texto = []
    texto.append(f"Probabilidad estimada de diabetes tipo 2: {prob:.2f}")
    texto.append(f"Categoría de riesgo: {riesgo_cat}")
    texto.append(f"Clasificación del modelo: {'Sí diabetes (1)' if clase == 1 else 'No diabetes (0)'}")

    texto.append("\nFactores que más influyen en tu riesgo:")

    indices = np.argsort(np.abs(shap_values))[::-1]
    for i in indices[:4]:
        nombre = feature_names[i]
        valor = shap_values[i]
        if valor > 0:
            texto.append(f" - {nombre}: aumenta el riesgo (SHAP = {valor:.3f})")
        else:
            texto.append(f" - {nombre}: reduce el riesgo (SHAP = {valor:.3f})")

    texto.append("\nEste resultado es cribado, no diagnóstico.")
    return "\n".join(texto)

# ============================================================
# 5. GUI Tkinter
# ============================================================

class CribadoDiabetesApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cribado de diabetes tipo 2 (RL + SHAP)")
        self.geometry("650x650")

        self.inputs = {}
        self._crear_formulario()
        self._crear_botones()
        self._crear_area_resultados()

    def _crear_formulario(self):
        frame = ttk.LabelFrame(self, text="Datos para cribado (variables de RFE)")
        frame.pack(fill="x", padx=10, pady=10)

        for i, col in enumerate(columnas_rl):
            nombre_es = mapa_es[col]
            ttk.Label(frame, text=nombre_es).grid(row=i, column=0, sticky="w", padx=5, pady=5)

            if col in opciones_combobox:
                valores_texto = list(opciones_combobox[col].values())
                combo = ttk.Combobox(frame, values=valores_texto, state="readonly")
                combo.grid(row=i, column=1, sticky="ew", padx=5, pady=5)
                combo.current(0)
                self.inputs[col] = combo
            else:
                entry = ttk.Entry(frame)
                entry.grid(row=i, column=1, sticky="ew", padx=5, pady=5)
                entry.insert(0, "0")
                self.inputs[col] = entry

    def _crear_botones(self):
        frame = ttk.Frame(self)
        frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(frame, text="Calcular riesgo", command=self._on_predict).pack(side="left", padx=5)
        ttk.Button(frame, text="Ver gráfico SHAP", command=self._on_show_shap).pack(side="left", padx=5)

    def _crear_area_resultados(self):
        frame = ttk.LabelFrame(self, text="Resultado y explicación")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.text_result = tk.Text(frame, wrap="word", height=20)
        self.text_result.pack(fill="both", expand=True, padx=5, pady=5)

    def _leer_valores(self):
        valores = {}
        try:
            for col, widget in self.inputs.items():
                if isinstance(widget, ttk.Combobox):
                    texto = widget.get()
                    for codigo, descripcion in opciones_combobox[col].items():
                        if descripcion == texto:
                            valores[col] = float(codigo)
                            break
                else:
                    val_str = widget.get().strip()
                    if val_str == "":
                        raise ValueError(f"El campo {mapa_es[col]} está vacío.")
                    valores[col] = float(val_str)
        except ValueError as e:
            messagebox.showerror("Error de entrada", str(e))
            return None
        return valores

    def _on_predict(self):
        valores = self._leer_valores()
        if valores is None:
            return

        prob, clase, shap_values, fnames = predecir_y_explicar(valores)
        texto = generar_texto_explicativo(prob, clase, shap_values, fnames)

        self.text_result.delete("1.0", tk.END)
        self.text_result.insert(tk.END, texto)

        self._last_shap_values = shap_values
        self._last_feature_names = fnames

    def _on_show_shap(self):
        if not hasattr(self, "_last_shap_values"):
            messagebox.showinfo("Info", "Primero calcula una predicción.")
            return
        mostrar_grafico_shap(self._last_shap_values, self._last_feature_names)


if __name__ == "__main__":
    app = CribadoDiabetesApp()
    app.mainloop()

