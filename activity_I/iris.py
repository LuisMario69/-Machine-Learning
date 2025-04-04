# Paso 7: Correlación entre características
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos Iris desde seaborn
data = sns.load_dataset('iris')

# Calcular la matriz de correlación SOLO con columnas numéricas
print("Calculando y visualizando matriz de correlación...")
correlation_matrix = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].corr()

# Mostrar la matriz de correlación con un mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Mapa de calor de la correlación entre características")
plt.savefig('correlation_heatmap.png')  # Guardar la figura
plt.show()