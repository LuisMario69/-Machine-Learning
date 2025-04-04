import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 1. Regresión Logística
def modelo_regresion_logistica():
    from sklearn.linear_model import LogisticRegression
    
    print("MODELO: REGRESIÓN LOGÍSTICA")
    print("=" * 50)
    # Cargar un conjunto de datos (Iris en este caso)
    data = load_iris()
    X = data.data
    y = data.target
    # Convertimos a un problema binario (0: Setosa, 1: No Setosa)
    y = (y == 0).astype(int)
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Crear y entrenar el modelo
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    print(f"Precisión: {accuracy_score(y_test, y_pred)}")
    print("\n")

# 2. Árboles de Decisión
def modelo_arbol_decision():
    from sklearn.tree import DecisionTreeClassifier
    
    print("MODELO: ÁRBOL DE DECISIÓN")
    print("=" * 50)
    # Cargar el conjunto de datos Iris
    data = load_iris()
    X = data.data
    y = data.target
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Crear el modelo de árbol de decisión
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    print(f"Precisión: {accuracy_score(y_test, y_pred)}")
    print("\n")

# 3. Máquinas de Soporte Vectorial (SVM)
def modelo_svm():
    from sklearn.svm import SVC
    
    print("MODELO: MÁQUINAS DE SOPORTE VECTORIAL (SVM)")
    print("=" * 50)
    # Cargar el conjunto de datos Iris
    data = load_iris()
    X = data.data
    y = data.target
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Crear el modelo SVM
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    print(f"Precisión: {accuracy_score(y_test, y_pred)}")
    print("\n")

# 4. K-Vecinos más Cercanos (KNN)
def modelo_knn():
    from sklearn.neighbors import KNeighborsClassifier
    
    print("MODELO: K-VECINOS MÁS CERCANOS (KNN)")
    print("=" * 50)
    # Cargar el conjunto de datos Iris
    data = load_iris()
    X = data.data
    y = data.target
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Crear el modelo KNN
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    print(f"Precisión: {accuracy_score(y_test, y_pred)}")
    print("\n")

# 5. Redes Neuronales Artificiales (ANN)
def modelo_red_neuronal():
    from sklearn.neural_network import MLPClassifier
    
    print("MODELO: RED NEURONAL ARTIFICIAL (ANN)")
    print("=" * 50)
    # Cargar el conjunto de datos Iris
    data = load_iris()
    X = data.data
    y = data.target
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Crear el modelo de red neuronal (MLP)
    model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    print(f"Precisión: {accuracy_score(y_test, y_pred)}")
    print("\n")

# 6. Random Forest (Bosque Aleatorio)
def modelo_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    
    print("MODELO: RANDOM FOREST (BOSQUE ALEATORIO)")
    print("=" * 50)
    # Cargar el conjunto de datos Iris
    data = load_iris()
    X = data.data
    y = data.target
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Crear el modelo Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    print(f"Precisión: {accuracy_score(y_test, y_pred)}")
    print("\n")

if __name__ == "__main__":
    print("ANÁLISIS DE MODELOS DE MACHINE LEARNING")
    print("="*50)
    print("Este script ejecuta 6 modelos diferentes de Machine Learning")
    print("utilizando el conjunto de datos Iris para mostrar su implementación.")
    print("\n")
    
    # Ejecutar todos los modelos
    modelo_regresion_logistica()
    modelo_arbol_decision()
    modelo_svm()
    modelo_knn()
    modelo_red_neuronal()
    modelo_random_forest()
    
    print("COMPARACIÓN DE MODELOS")
    print("="*50)
    print("Regresión logística: Útil para clasificación binaria.")
    print("Árboles de decisión: Fáciles de interpretar, útiles para clasificación y regresión.")
    print("SVM: Eficaz para problemas complejos, especialmente con datos no lineales.")
    print("KNN: Modelo simple y efectivo basado en la proximidad.")
    print("Redes neuronales: Adecuadas para problemas complejos y grandes cantidades de datos.")
    print("Random Forest: Modelo robusto y preciso que combate el sobreajuste.")