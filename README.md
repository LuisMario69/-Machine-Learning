# Actividad 2 - Modelos de Machine Learning

Este repositorio contiene la implementación de varios modelos de Machine Learning utilizando el conjunto de datos Iris. (Tiene una carpeta llamada  activity_I que son las gráficas de barras, histogramas, etc.)

## 📁 Estructura del repositorio

- `modelos_ml.py` - Implementación de 6 modelos diferentes de clasificación: **Regresión Logística, Árboles de Decisión, SVM, KNN, Redes Neuronales y Random Forest**.
- `regresion_lineal.py` - Implementación paso a paso de un modelo de **regresión lineal**.
- `informe_resultados.md` - Análisis detallado de los **resultados obtenidos**.
- `requirements.txt` - Lista de dependencias necesarias para ejecutar los scripts.
- `README.md` - Este archivo con la descripción general del proyecto.

## ⚙️ Requisitos previos

Para ejecutar los scripts de este proyecto, necesitas tener instalado **Python 3.6 o superior** y las siguientes bibliotecas:

```txt
pandas>=1.0.0
numpy>=1.18.0
matplotlib>=3.1.0
seaborn>=0.10.0
scikit-learn>=0.22.0
```

Puedes instalar todas las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

## 🚀 Ejecución de los scripts

### 🔹 Para ejecutar los modelos de clasificación:

```bash
python modelos_ml.py
```

Este script implementará y evaluará los **6 modelos de clasificación** utilizando el conjunto de datos **Iris**.

### 🔹 Para ejecutar el modelo de regresión lineal:

```bash
python regresion_lineal.py
```

Este script realizará un análisis completo de **regresión lineal** utilizando el conjunto de datos **Iris** para predecir la longitud del pétalo.

## 📊 Resultados

Los resultados detallados de la ejecución de los modelos se encuentran en el archivo [`informe_resultados.md`](informe_resultados.md). 

Además, el script de **regresión lineal** generará un gráfico llamado `predicciones_vs_real.png` que muestra la comparación entre los valores reales y las predicciones del modelo.

## 👤 Autor

Luis Mario Vargas A
Anderson Blandon

## 📜 Licencia

Este proyecto está licenciado bajo la **Licencia MIT** - ver el archivo [`LICENSE`](LICENSE) para más detalles.
