# NeumoniaDetection
En este trabajo se presenta una herramienta para la detección de neumonía en radiografías de rayos-X, utilizando aprendizaje automático, esta herramienta es un software que está programada totalmente en Python.
Para realizar este proyecto, se descargaron bancos de imágenes de radiografías de rayos-X a las cuales se les aplica un preprocesamiento para estandarizar las imágenes, este consiste en la mejora de contraste aplicando ecualización del histograma, además las imágenes son recortadas de tal manera que se obtenga únicamente el área de los pulmones y se redimensionan a un tamaño de 256x256.
Se aplica análisis de componentes principales (PCA), los pesos resultantes de la proyección de las imágenes sobre los componentes principales se normalizan y se seleccionan las características que mejor separan las dos clases. 
Estás características se utilizan para entrenar un algoritmo de generación/selección de prototipos propuesto para que represente de mejor manera el conjunto de datos de entrenamiento, esto al entregar solo los prototipos que aporten información relevante sobre la distribución del conjunto de datos, como es el caso de las muestras que se encuentran en la frontera.
Este sistema produce como resultados un estado normal o la probabilidad de presencia de neumonía.
La interfaz fue creada con Tkinter para facilitar la interacción con un usuario médico. 
Para el programa se utilizaron las librerías numpy, openCV, os y math.
