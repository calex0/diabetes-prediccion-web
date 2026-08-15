[TOC]



## **Crear un entorno con Python 3.9**

Regresión logística:

- Ubuntu 22.04
- Python 3.9
- CPU‑only



```bash
cd ~/diab![1-GUI_y_prediccion](/home/alex/Documentos/diabetesPrediccionApp4-rl/capturas_app_GUI/1-GUI_y_prediccion.png)etes-prediccion-main/app
```



Antes de crear un entorno env hay que instalar Python 3.9. En Ubuntu 26.04 hay que hacer lo siguiente:

```bash
sudo apt update
sudo apt install software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install python3.9 python3.9-venv python3.9-dev python3.9-tk
```
Ahora creamos el entorno python:
```bash
***n Ubuntu 26.04
pyenv virtualenv 3.9.18 app_diabetes_env
pyenv activate app_diabetes_env
pip install -r requirements.txt
```

## Ejemplo de ejecución de la app de cribado

![](/home/alex/Documentos/diabetesPrediccionPortafolio/diabetes_app/Imagen1.png)

![](/home/alex/Documentos/diabetesPrediccionPortafolio/diabetes_app/Imagen2.png)

## Significado de las variables de cribado de la app

Variable    Descripción

CTELENUM    Número de teléfonos en el hogar.
ASBIRDUC    Consumo de bebidas dietéticas.
WTCHSALT    Evita la sal en las comidas.
SCNTMEL1    Frecuencia de sentirse solo.
CTELNUM1    Uso del teléfono celular.
_RFHLTH     Estado general de salud percibida.
RACEG21     Categoría racial / étnica.
_AGE_G      Grupo de edad.
_BMI5CAT    Categoría de IMC (Índice de Masa Corporal).
