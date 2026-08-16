import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap


st.markdown("---")
st.info("💡 **Información importante sobre el uso de esta herramienta**")
st.caption(
    "**Descargo de responsabilidad:** El resultado proporcionado por este modelo es una estimación estadística "
    "y **no debe ser utilizado como un diagnóstico médico definitivo.** Las decisiones relacionadas con la salud, "
    "cambios en el estilo de vida o tratamientos deben tomarse siempre bajo la supervisión de un médico colegiado. "
    "El desarrollador de esta plataforma no se hace responsable del uso de la información aquí presentada."
)

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

columnas_rl = joblib.load("columnas_rl.pkl")
FEATURE_NAMES = [mapa_es[col] for col in columnas_rl]

# ============================================================
# 2. Opciones de los combobox (BRFSS)
# ============================================================

opciones_combobox = {
    "_RFHLTH": {0: "Good/Excellent", 1: "Fair", 2: "Poor"},
    "_AGE_G": {0: "18-24", 1: "25-34", 2: "35-44", 3: "45-54", 4: "55-64", 5: "65+"},
    "_BMI5CAT": {0: "Underweight", 1: "Normal", 2: "Overweight", 3: "Obese"},
    "_RACEG21": {0: "White", 1: "Black", 2: "Other"},
    "WTCHSALT": {0: "No", 1: "Yes", 2: "Don't know", 3: "Refused"},
    "ASBIRDUC": {0: "No", 1: "Yes", 2: "Don't know", 3: "Refused"},
    "CTELNUM1": {0: "None", 1: "One phone", 2: "Two or more"},
    "CTELENUM": {0: "No", 1: "Yes", 2: "Don't know"},
    "SCNTMEL1": {
        0: "0 days", 1: "1–7 days", 2: "8–14 days", 3: "15–21 days",
        4: "22–30 days", 5: "31–60 days", 6: "61–90 days", 7: "90+ days"
    }
}

# ============================================================
# 3. Cargar pipeline RL + SHAP
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
        data[col] = [float(valores_dict.get(col, 0.0))]
    return pd.DataFrame(data)


def agrupar_shap_por_variable_original(shap_values_full):
    shap_por_var = {}
    for col in columnas_rl:
        cols_var = [i for i, name in enumerate(columnas_transformadas) if col in name]
        shap_por_var[col] = shap_values_full[cols_var].sum() if cols_var else 0.0
    return np.array([shap_por_var[col] for col in columnas_rl])


def predecir_y_explicar(valores_dict):
    x = construir_vector_entrada(valores_dict)
    prob = pipeline.predict_proba(x)[0, 1]
    clase = int(pipeline.predict(x)[0])

    x_trans = preproc.transform(x)
    shap_values_full = explainer.shap_values(x_trans)[0]
    shap_values_rfe = agrupar_shap_por_variable_original(shap_values_full)

    return prob, clase, shap_values_rfe, FEATURE_NAMES


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


def mostrar_grafico_shap(shap_values, feature_names):
    fig, ax = plt.subplots(figsize=(7, 4))
    indices = np.argsort(np.abs(shap_values))[::-1]
    ordered_features = [feature_names[i] for i in indices]
    ordered_shap = shap_values[indices]
    colors = ["red" if v > 0 else "blue" for v in ordered_shap]

    ax.barh(ordered_features[::-1], ordered_shap[::-1], color=colors[::-1])
    ax.set_xlabel("Valor SHAP (contribución al riesgo)")
    ax.set_title("Explicación SHAP de la predicción")
    plt.tight_layout()
    return fig

# ============================================================
# 5. Interfaz Streamlit
# ============================================================

st.title("Cribado de diabetes tipo 2 (RL + SHAP)")
st.markdown("Herramienta de cribado basada en regresión logística y explicabilidad SHAP.")

st.header("Datos para cribado")

valores = {}

for col in columnas_rl:
    nombre_es = mapa_es[col]

    if col in opciones_combobox:
        opciones = opciones_combobox[col]
        texto = st.selectbox(nombre_es, list(opciones.values()))
        codigo = [k for k, v in opciones.items() if v == texto][0]
        valores[col] = float(codigo)
    else:
        valores[col] = float(st.number_input(nombre_es, value=0.0))

if st.button("Calcular riesgo"):
    prob, clase, shap_values, fnames = predecir_y_explicar(valores)
    texto = generar_texto_explicativo(prob, clase, shap_values, fnames)
    st.text(texto)
    st.session_state["shap_values"] = shap_values
    st.session_state["feature_names"] = fnames

if st.button("Ver gráfico SHAP"):
    if "shap_values" not in st.session_state:
        st.info("Primero calcula una predicción.")
    else:
        fig = mostrar_grafico_shap(
            st.session_state["shap_values"],
            st.session_state["feature_names"]
        )
        st.pyplot(fig)

