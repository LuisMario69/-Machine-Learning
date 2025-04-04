import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def regresion_lineal_iris():
    print("IMPLEMENTACIÓN DE REGRESIÓN LINEAL CON EL CONJUNTO DE DATOS IRIS")
    print("="*70)
    
    # Paso 1: Cargar el conjunto de datos
    print("Paso 1: Cargando el conjunto de datos Iris")
    data = sns.load_dataset('iris')
    print("Primeras filas del conjunto de datos:")
    print(data.head())
    print("\n")
    
    # Paso 2: Preparación de los datos
    print("Paso 2: Preparando los datos")
    # Seleccionamos las características (features) y la variable objetivo (target)
    X = data[['sepal_length', 'sepal_width', 'petal_width']]  # Variables predictoras
    y = data['petal_length']  # Variable objetivo
    
    # Dividir el conjunto de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Tamaño del conjunto de entrenamiento:", X_train.shape)
    print("Tamaño del conjunto de prueba:", X_test.shape)
    print("\n")
    
    # Paso 3: Creación del modelo de regresión lineal
    print("Paso 3: Creando y entrenando el modelo de regresión lineal")
    # Crear el modelo de regresión lineal
    model = LinearRegression()
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Ver los coeficientes y la intersección (intercepto) del modelo
    print(f"Coeficientes: {model.coef_}")
    print(f"Intersección (intercepto): {model.intercept_}")
    print("\n")
    
    # Paso 4: Hacer predicciones
    print("Paso 4: Realizando predicciones")
    # Realizar predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Mostrar las predicciones y los valores reales
    predictions_df = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
    print("Comparación de valores reales vs. predicciones:")
    print(predictions_df.head())
    print("\n")
    
    # Paso 5: Evaluación del modelo
    print("Paso 5: Evaluando el modelo")
    # Calcular el error cuadrático medio (MSE) y R²
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Error cuadrático medio (MSE): {mse}")
    print(f"Coeficiente de determinación R²: {r2}")
    print("\n")
    
    # Paso 6: Visualización de resultados
    print("Paso 6: Visualizando resultados")
    print("Generando gráfico de valores reales vs. predicciones...")
    # Graficar los valores reales vs. las predicciones
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicciones')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             color='red', linewidth=2, label="Línea de referencia")
    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.title("Valores reales vs Predicciones")
    plt.legend()
    plt.savefig('predicciones_vs_real.png')  # Guardar gráfico
    plt.close()
    print("Gráfico guardado como 'predicciones_vs_real.png'")
    print("\n")
    
    # Paso 7: Conclusiones del modelo
    print("Paso 7: Conclusiones del modelo")
    print("- Los coeficientes de la regresión lineal nos indican cómo cada característica")
    print("  impacta en la predicción de la longitud del pétalo (petal_length).")
    print("- El error cuadrático medio (MSE) nos da una idea de cuán precisas son nuestras")
    print("  predicciones.")
    print("- El coeficiente de determinación R² indica qué tan bien el modelo está ajustado")
    print("  a los datos. Un valor cercano a 1 significa que el modelo explica bien la")
    print("  variabilidad de la variable objetivo.")
    
    # Interpretación de los coeficientes
    features = ['sepal_length', 'sepal_width', 'petal_width']
    for i, feature in enumerate(features):
        print(f"- Un aumento de 1 unidad en {feature} está asociado con un cambio")
        print(f"  de {model.coef_[i]:.4f} unidades en la longitud del pétalo.")
    
    if r2 > 0.8:
        print("- Con un R² de {:.4f}, el modelo tiene un buen ajuste a los datos.".format(r2))
    elif r2 > 0.5:
        print("- Con un R² de {:.4f}, el modelo tiene un ajuste moderado a los datos.".format(r2))
    else:
        print("- Con un R² de {:.4f}, el modelo tiene un ajuste débil a los datos.".format(r2))

if __name__ == "__main__":
    regresion_lineal_iris()