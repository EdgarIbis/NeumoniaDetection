# NeumoniaDetection
En este trabajo se presenta una herramienta para la detección de neumonía en radiografías de rayos-X, utilizando aprendizaje automático, esta herramienta es un software que está programada totalmente en Python.

Para realizar este proyecto, en la etapa de entrenamiento, se descargaron bancos de imágenes de radiografías de rayos-X a las cuales se les aplicó un preprocesamiento para estandarizar las imágenes, este consiste en la mejora de contraste aplicando ecualización del histograma, además las imágenes son recortadas de tal manera que se obtenga únicamente el área de los pulmones y se redimensionan a un tamaño de 256x256.

Se aplica análisis de componentes principales (PCA), los pesos resultantes de la proyección de las imágenes sobre los componentes principales se normalizan y se seleccionan las características que mejor separan las dos clases. Estás características se utilizan para entrenar un algoritmo de generación/selección de prototipos propuesto para que represente de mejor manera el conjunto de datos de entrenamiento, esto al entregar solo los prototipos que aporten información relevante sobre la distribución del conjunto de datos, como es el caso de las muestras que se encuentran en la frontera.

Este sistema produce como resultados un estado normal o la probabilidad de presencia de neumonía.

La interfaz fue creada con Tkinter para facilitar la interacción con un usuario médico. 

Para el programa se utilizaron las librerías numpy, openCV, os y math.

## Como usar
El programa principal tiene el nombre de Interfaz Gráfica, solo se debe ejecutar este programa, los demás archivos sirven para que la interfaz se ejecute correctamente. 

El sistema aplica k-NN ponderado tradicional utilizando como prototipos los generados en la etapa de entrenamiento. Si la etiqueta predicha es normal, se despliega este resultado en la ventana de interfaz. Por el contrario, si la etiqueta estimada es presencia de neumonía se realiza un cálculo de probabilidad de neumonía tomando en cuenta la votación que arrojó cada clase:

$$
P = \frac{v1}{v0+v1} * 100
$$

donde v1 representa la votación de la clase neumonía y v0 es el valor de la votación de la clase normal. De esta manera, se obtendrán valores de probabilidad mayores a 50 que
dependen de los vecinos que rodean a la muestra a clasificar.
