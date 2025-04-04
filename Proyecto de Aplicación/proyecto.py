# Proyecto Simplificado de Predicción: Clasificación de Especies de Iris
# Fecha: Abril 2025

# Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris

# Cargar los datos
print("Cargando conjunto de datos Iris...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
species_names = iris.target_names

# Crear DataFrame completo
df = X.copy()
df['species'] = y
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Mostrar información básica
print(f"Tamaño del conjunto de datos: {df.shape}")
print(f"Distribución de especies:\n{df['species_name'].value_counts()}")

# 1. Primera gráfica: Diagrama de dispersión de características
print("\nGenerando gráfica 1: Diagrama de dispersión...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', 
               hue='species_name', data=df, palette='viridis')
plt.title('Iris: Longitud vs Ancho del Pétalo')
plt.savefig('grafica1_scatter.png')
plt.show()

# 2. Segunda gráfica: Diagrama de cajas para ver la distribución
print("\nGenerando gráfica 2: Diagrama de cajas...")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='species_name', y='petal length (cm)', data=df)
plt.title('Longitud del Pétalo por Especie')

plt.subplot(1, 2, 2)
sns.boxplot(x='species_name', y='petal width (cm)', data=df)
plt.title('Ancho del Pétalo por Especie')
plt.tight_layout()
plt.savefig('grafica2_boxplot.png')
plt.show()

# Preparar datos para entrenamiento
print("\nPreparando datos para entrenamiento...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir modelos a evaluar
print("\nEntrenando modelos...")
modelos = {
    'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
    'Árbol de Decisión': DecisionTreeClassifier(random_state=42)
}

# Resultados
resultados = {}

# Entrenar y evaluar cada modelo
for nombre, modelo in modelos.items():
    # Entrenar el modelo
    modelo.fit(X_train_scaled, y_train)
    
    # Realizar predicciones
    y_pred = modelo.predict(X_test_scaled)
    
    # Para clasificación: Calcular precisión
    accuracy = accuracy_score(y_test, y_pred)
    
    # Para regresión: Calcular RMSE
    # Convertimos las predicciones a valores continuos para simular un problema de regresión
    # Esto es solo para demostrar el cálculo de RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Guardar resultados
    resultados[nombre] = {
        'accuracy': accuracy,
        'rmse': rmse
    }
    
    # Imprimir resultados
    print(f"\n===== Modelo: {nombre} =====")
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print(f"Error Cuadrático Medio (RMSE): {rmse:.4f}")
    print("\nMatriz de confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred, target_names=species_names))

# Comparar modelos gráficamente
print("\nGenerando gráfica comparativa de modelos...")
nombres = list(resultados.keys())
accuracy_values = [resultados[nombre]['accuracy'] for nombre in nombres]
rmse_values = [resultados[nombre]['rmse'] for nombre in nombres]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Eje para Accuracy
color = 'tab:blue'
ax1.set_xlabel('Modelo')
ax1.set_ylabel('Precisión (Accuracy)', color=color)
ax1.bar([x - 0.2 for x in range(len(nombres))], accuracy_values, 0.4, label='Accuracy', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(range(len(nombres)))
ax1.set_xticklabels(nombres)

# Eje para RMSE
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('RMSE', color=color)
ax2.bar([x + 0.2 for x in range(len(nombres))], rmse_values, 0.4, label='RMSE', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Comparación de Modelos: Precisión vs RMSE')
plt.tight_layout()
plt.savefig('comparacion_modelos.png')
plt.show()

print("\n=== PROYECTO COMPLETADO ===")