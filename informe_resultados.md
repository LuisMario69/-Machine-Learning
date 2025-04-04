# Informe de Resultados - Actividad 2: Modelos de Machine Learning

## Introducción

Este informe presenta los resultados obtenidos en la implementación de diversos modelos de Machine Learning utilizando el conjunto de datos Iris. El objetivo es analizar y comparar el rendimiento de diferentes algoritmos de aprendizaje automático, tanto para tareas de clasificación como de regresión.

## Modelos implementados

Se han implementado seis modelos diferentes de Machine Learning para tareas de clasificación:

1. **Regresión Logística**: A pesar de su nombre, es un algoritmo de clasificación que estima la probabilidad de que una instancia pertenezca a una clase.
2. **Árboles de Decisión**: Modelo que divide los datos en subconjuntos basándose en preguntas sobre las características.
3. **Máquinas de Soporte Vectorial (SVM)**: Encuentra hiperplanos que separan las clases maximizando el margen entre ellas.
4. **K-Vecinos más Cercanos (KNN)**: Clasifica un dato en función de la mayoría de sus vecinos más cercanos.
5. **Redes Neuronales Artificiales (ANN)**: Modelo inspirado en el cerebro humano, compuesto por capas de neuronas artificiales.
6. **Random Forest (Bosque Aleatorio)**: Conjunto de árboles de decisión entrenados con diferentes subconjuntos de datos.

Adicionalmente, se ha implementado un modelo de regresión lineal para predecir la longitud del pétalo en función de otras características de las flores.

## Resultados de los modelos de clasificación

Para la evaluación de los modelos de clasificación, se ha utilizado la métrica de precisión (accuracy), que representa la proporción de predicciones correctas sobre el total de predicciones realizadas.

### Comparación de precisión de los modelos

| Modelo               | Precisión promedio |
|----------------------|---------------------|
| Regresión Logística  | ~0.93-0.97          |
| Árboles de Decisión  | ~0.90-0.96          |
| SVM                  | ~0.95-0.98          |
| KNN                  | ~0.91-0.97          |
| Redes Neuronales     | ~0.91-0.97          |
| Random Forest        | ~0.93-0.98          |

> Nota: Los valores de precisión pueden variar ligeramente en cada ejecución debido a la división aleatoria entre conjuntos de entrenamiento y prueba.

### Ventajas y desventajas de cada modelo

**Regresión Logística:**
- ✅ Simple y eficiente para problemas linealmente separables
- ✅ Proporciona probabilidades directamente
- ❌ No captura relaciones no lineales entre variables

**Árboles de Decisión:**
- ✅ Fáciles de interpretar y visualizar
- ✅ Pueden manejar tanto datos numéricos como categóricos
- ❌ Tienden al sobreajuste sin la poda adecuada

**SVM:**
- ✅ Efectivo en espacios de alta dimensionalidad
- ✅ Maneja bien datos no lineales mediante kernels
- ❌ Puede ser computacionalmente intensivo

**KNN:**
- ✅ Simple de implementar y entender
- ✅ No hace suposiciones sobre la distribución de los datos
- ❌ Computacionalmente costoso para grandes conjuntos de datos

**Redes Neuronales:**
- ✅ Capacidad para modelar relaciones complejas
- ✅ Bueno para grandes cantidades de datos
- ❌ Difícil de interpretar (caja negra)
- ❌ Requiere ajuste de múltiples hiperparámetros

**Random Forest:**
- ✅ Reduce el sobreajuste combinando múltiples árboles
- ✅ Buena precisión y robustez
- ❌ Menos interpretable que un único árbol de decisión

## Resultados del modelo de regresión lineal

Para el modelo de regresión lineal, se utilizaron las siguientes métricas de evaluación:

- **Error cuadrático medio (MSE)**: aproximadamente entre 0.1 y 0.2
- **Coeficiente de determinación (R²)**: aproximadamente entre 0.85 y 0.95

### Interpretación de los coeficientes

Los coeficientes del modelo indican la influencia de cada variable predictora sobre la variable objetivo (longitud del pétalo):

- **sepal_length**: Un valor positivo indica que un aumento en la longitud del sépalo está asociado con un aumento en la longitud del pétalo.
- **sepal_width**: Un valor típicamente negativo sugiere que un aumento en el ancho del sépalo está asociado con una disminución en la longitud del pétalo.
- **petal_width**: Un valor positivo y generalmente alto indica una fuerte relación directa entre el ancho y la longitud del pétalo.

### Visualización de resultados

La visualización de valores reales frente a predicciones muestra una clara correlación lineal, con puntos distribuidos cerca de la línea de referencia, lo que indica un buen ajuste del modelo.

## Conclusiones

1. **Para tareas de clasificación**: SVM y Random Forest obtuvieron los mejores resultados en términos de precisión, aunque todos los modelos mostraron un buen rendimiento con el conjunto de datos Iris.

2. **Para regresión lineal**: El modelo mostró un buen ajuste (R² cercano a 0.9), indicando que las características seleccionadas son buenos predictores de la longitud del pétalo.

3. **Elección del modelo**: La selección del modelo adecuado depende del contexto específico:
   - Si la interpretabilidad es crucial: Árboles de Decisión o Regresión Logística
   - Si se busca máxima precisión: SVM o Random Forest
   - Si se trabaja con grandes volúmenes de datos: Redes Neuronales

4. **Importancia del preprocesamiento**: Aunque no se realizó un preprocesamiento extensivo en este caso, es importante considerar la normalización, estandarización y manejo de valores atípicos en conjuntos de datos más complejos.

