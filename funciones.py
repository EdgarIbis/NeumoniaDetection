# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:17:22 2019
Conjunto de funciones utilizadas para implementar el algoritmo de eigenfaces
Las funciones son las siguientes:
    Cargar_imag(ruta, numIm)
        Esta funcion recibe como parametros la ruta de donde se van a extraer
        las imagenes y el numero de imagenes que se desean cargar.
        Entrega una matriz cuya columna representa una imagen, es decir cada
        imagen se almacena como vector columna
    show_im(vec_im)
        Recibe como parametros una imagen en forma de vector columna y la 
        convierte en matriz con las dimensiones de la imagen para poder ser 
        mostrada. Hace un recorrido en el vector para asegurarse que no haya
        valores negativos en el vector ocasionados por impresiciones en los
        calculos para poder hacer una correcta conversion a uint8. Esta
        funcion muestra la imagen en escala de grises.
    resta_psi(matriz, prom)
        Recibe como parametros la matriz de imagenes columna y el vector 
        promedio de la matriz de imagenes. La funcion resta a cada imagen 
        columna el vector promedio.
        Entrega la matriz de imagenes columna menos la resta del promedio.
    k_98(valproord, numIm)
        Recibe los valores propios ordenados de mayor a menor y entrega el
        numero de eigenfaces necesarios para llegar al 98% de representación.
    normalizacion(ui, numIm)
        Recibe la matriz ui y el numero de imagenes. Entrega ui normalizado.
    ponderado(ui, A, numIm)
        Recibe ui, la matriz A y el numero de imagenes para calcular los pesos
        de cada imagen dentro del banco. Entrega los pesos
        de cada imagen cargada.

@author: Edgar I.T.M
"""

import os
import cv2
import math
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt

#FUNCION PARA CARGAR IMAGENES
def cargar_imag(ruta, numIm):
    print("Leyendo imagenes de " + ruta, end = "...", flush=True)
    directorio = sorted(os.listdir(ruta))
    imagen = cv2.imread(os.path.join(ruta, directorio[1]))#para obtener la dim
    [filas, columnas, r] = imagen.shape
    [filas, columnas] = int(filas/2),int(columnas/2)
    print(filas,columnas)
    vec_imag = np.empty((filas*columnas, numIm))
    for i in range(numIm):
        imPath = os.path.join(ruta, directorio[i])
        im = cv2.imread(imPath) 
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) #convierte a gris
        im = cv2.resize(im, (columnas,filas))
        tempo = np.reshape(im,(filas*columnas))
        vec_imag[:,i] = tempo
    return vec_imag

#FUNCION PARA CARGAR IMAGENES
def cargar_imagDiv(ruta, numIm, div):
    print("Leyendo imagenes de " + ruta, end = "...", flush=True)
    directorio = sorted(os.listdir(ruta))
    imagen = cv2.imread(os.path.join(ruta, directorio[1]))#para obtener la dim
    [filas, columnas, r] = imagen.shape
    [filas, columnas] = int(filas/div),int(columnas/div)
    print(filas,columnas)
    vec_imag = np.empty((filas*columnas, numIm))
    for i in range(numIm):
        imPath = os.path.join(ruta, directorio[i])
        im = cv2.imread(imPath) 
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) #convierte a gris
        im = cv2.resize(im, (columnas,filas))
        tempo = np.reshape(im,(filas*columnas))
        vec_imag[:,i] = tempo
    return vec_imag

#FUNCION PARA CARGAR IMAGENES CLAHE, REDIM, 256*256
def cargar_imag_CLAHE(ruta, numIm):
    print("Leyendo imagenes de " + ruta, end = "...", flush=True)
    directorio = sorted(os.listdir(ruta))
    # imagen = cv2.imread(os.path.join(ruta, directorio[1]))#para obtener la dim
    clahe = cv2.createCLAHE()
    # [filas, columnas, r] = imagen.shape1
    # [filas, columnas] = int(filas/div),int(columnas/div)
    filas,columnas = 256,256
    print(filas,columnas)
    vec_imag = np.empty((filas*columnas, numIm))
    for i in range(numIm):
        imPath = os.path.join(ruta, directorio[i])
        im = cv2.imread(imPath) 
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) #convierte a gris
        im = clahe.apply(im)
        im = cv2.resize(im, (columnas,filas))
        tempo = np.reshape(im,(filas*columnas))
        vec_imag[:,i] = tempo
    return vec_imag

#FUNCION PARA CARGAR IMAGENES EQUHIS, REDIM, 256*256
def cargar_imag_equHist(ruta, numIm):
    print("Leyendo imagenes de " + ruta, end = "...", flush=True)
    directorio = sorted(os.listdir(ruta))
    filas,columnas = 256,256
    print(filas,columnas)
    vec_imag = np.empty((filas*columnas, numIm))
    for i in range(numIm):
        imPath = os.path.join(ruta, directorio[i])
        im = cv2.imread(imPath) 
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) #convierte a gris
        im = cv2.equalizeHist(im)
        im = cv2.resize(im, (columnas,filas))
        tempo = np.reshape(im,(filas*columnas))
        vec_imag[:,i] = tempo
    return vec_imag

#FUNCION PARA CARGAR IMAGENES REDIM, 256*256
def cargar_imag_square(ruta, numIm):
    print("Leyendo imagenes de " + ruta, end = "...", flush=True)
    directorio = sorted(os.listdir(ruta))
    filas,columnas = 256,256
    print(filas,columnas)
    vec_imag = np.empty((filas*columnas, numIm))
    for i in range(numIm):
        imPath = os.path.join(ruta, directorio[i])
        im = cv2.imread(imPath) 
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) #convierte a gris
        im = cv2.resize(im, (columnas,filas))
        tempo = np.reshape(im,(filas*columnas))
        vec_imag[:,i] = tempo
    return vec_imag

# CARGAR 1 SOLA IMAGEN
def cargar_1img(ruta, img):
    """
    

    Parameters
    ----------
    ruta : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    im : TYPE
        DESCRIPTION.

    """
    directorio = sorted(os.listdir(ruta))
    im = cv2.imread(os.path.join(ruta, directorio[img]))#para obtener la dim
    [filas, columnas, r] = im.shape
    [filas, columnas] = int(filas/2),int(columnas/2)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) #convierte a gris
    im = cv2.resize(im, (columnas,filas))
    return im
#FUNCION PARA MOSTRR IMAGENES
def show_im(vecim):
    for i in range(len(vecim)):
        if vecim[i] < 0:
            vecim[i] = 0
    im = np.reshape(vecim,(360,260))
    im = im.astype(np.uint8)
    plt.figure()
    plt.imshow(im,cmap='gray')
    plt.show()
def show_im2(vecim,fil,col):
    for i in range(len(vecim)):
        if vecim[i] < 0:
            vecim[i] = 0
    im = np.reshape(vecim,(col,fil))
    im = im.astype(np.uint8)
    plt.figure()
    plt.imshow(im,cmap='gray')
    plt.show()
#FUNCION PARA RESTAR IMAGENES    
def resta_psi(matriz, prom):
    dim = matriz.shape
    matprom = np.empty((dim[0],dim[1]))
    for j in range(dim[1]):
        matprom[:,j]=matriz[:,j] - prom
    return matprom
#FUNCION PARA ENCONTRAR EL NUMERO DE EIGENFACES NECESARIOS PARA OBTENER EL 98%
def k_98(valproord, numIm):
    eigsum = np.sum(valproord)
    csum = 0
    cociente = np.zeros(numIm)
    plt.figure()
    for l in range(numIm):
        csum = csum + valproord[l]  #suma de uno en uno los valores propios
        cociente[l] = csum/eigsum   #divide la suma uno a uno entre el total
        #if cociente[l] > 0.98 :     #es mayor a .98 rompe el ciclo
        if np.any(cociente > 0.98):
            k98 = l
            break
    plt.plot(cociente)
    return k98
#FUNCION DE NORMALIZACION
def normalizacion(ui, numIm):
    z = ui.shape
    uinorm = np.zeros((z[0],z[1]))
    for m in range(numIm):
        uinorm[:,m] = ui[:,m]/(math.sqrt(np.sum((ui[:,m])**2)))
    return uinorm
#FUNCION PARA OBTENER EL PONDERADO DE LAS IMAGENES DENTRO DEL BANCO
def ponderado(ui, A, numIm):
    wi = np.zeros((numIm,numIm))
    uit = ui.transpose()
    plt.figure()
    for n in range(numIm):
        for p in range(numIm):
            wi[n,p] = uit[p,:] @ A[:,n]
        #plt.ion()
        # plt.plot(wi[n,:])
    return wi


def funeigenfaces(ruta, numIm):
    """
    Carga imagenes del parametro ruta,calcula los eigenfaces de las imagenes de 
entrada. Las imagenes deben ser del mismo tamaño y el formato debe ser jpg.
    
Devuelve: eigenfaces principales normalizados, vector columna promedio, el
número de imágenes necesarios para obtener el 98% del total de la variación,
los pesos w de las imágenes cargadas.
    
Parámetros:
    Ruta: Es el directorio de donde se van a extraer la imágenes el formato es:
        ruta = 'C:/User/Desktop/caras/'  #Directorio
    numIm: Es el numero de imagenes que se desean cargar. Formato entero.
    
Bibliografía
M. Turk and A. Pentland, "Eigenfaces for Recognition", Journal of Cognitive 
Neuroscience, vol.3, no. 1, pp. 71-86, 1991.

    Parameters
    ----------
    ruta : TYPE
        DESCRIPTION.
    numIm : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    

        
    print("Leyendo imagenes de " + ruta, end = "...", flush=True)
    directorio = os.listdir(ruta)           #Directori de imagenes
    imagen = cv2.imread(os.path.join(ruta, directorio[1])) # se leé una imagen-
    [filas, columnas, r] = imagen.shape          #-para obtener su dimensión
    [filas, columnas] = int(filas/2),int(columnas/2)
    print(filas,columnas)
    vec_imag = np.zeros((filas*columnas, numIm)) #Matriz de almacenamiento
    for i in range(numIm):                       #Ciclo para almacenar imagenes
        imPath = os.path.join(ruta, directorio[i]) #Ruta de cada imagen
        im = cv2.imread(imPath)                    #Lee la imagen
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  #Convierte a escala de gris
        im = cv2.resize(im, (columnas,filas))
        tempo = np.reshape(im,(filas*columnas)) #Convierte imagen a vector col
        vec_imag[:,i] = tempo                  #Almacena la imagen en la matriz

    psi = np.mean(vec_imag, axis=1)             #Calcula el vector promedio
    print("IMAGEN PROMEDIO")                    #Mensaje
    improm = np.reshape(psi,(filas,columnas))          #Convierte vector a matriz
    improm = improm.astype(np.uint8)            #Convierte a entero sin signo
    plt.figure()
    plt.imshow(improm,cmap='gray')              #Convierte a escala de grises
    plt.show()                                  #Muestra la imagen promedio
    
    dim = vec_imag.shape           #Obtiene dimesión de la matriz de imágenes
    A = np.empty((dim[0],dim[1]))  #Crea una matriz vacia para almacenamiento
    for j in range(dim[1]):        #Ciclo para restar a cada imagen el promedio
        A[:,j] = vec_imag[:,j] - psi #Resta a cada imagen la imagen promedio
    
    matcov = A.T @ A             #Matriz de covarianza
    
    valpro, vecpro = la.eig(matcov) #eigenvectores y eigenvalores 
    valpro = np.real(valpro)    #Debido a imprecisión en ocasiones se obtienen-
    vecpro = np.real(vecpro)    #-valores complejos, se toma solo la parte real

    ind = valpro.argsort()     #obtiene los indices del vector de menor a mayor
    indice = ind[::-1]         #ordenar de los indices de mayor a menor (inver)
    valproord = valpro[indice] #eigenvalores ordenados
    plt.figure()               #Crea una nueva figura (ventana)
    plt.plot(valproord)        #ploteo de valores ordenados
    vi = vecpro[:,indice]      #eigenvectores ordenados de mayor a menor

    eigsum = np.sum(valproord) #Suma todos los eigenvalores
    csum = 0                   #Variable de almacenamiento temporal
    cociente = np.zeros(numIm) #Vector de almacenamiento
    plt.figure()               #Crea una nueva figura (ventana)
    for l in range(numIm):
        csum = csum + valproord[l]   #Suma de uno en uno los valores propios
        cociente[l] = csum/eigsum    #Sivide la suma uno a uno entre el total
        #if cociente[l] > 0.98 :     
        if np.any(cociente > 0.98):
            k98 = l                  #Si es mayor a .98 rompe el ciclo
            break
    plt.plot(cociente)               #Ploteo de la curva de crecicmiento

    ui = A @ vi;                     #Eigenfaces
    z = ui.shape                     #Dimensión de ui
    uin = np.zeros((z[0],z[1]))      #Matriz de almacenamiento
    for m in range(numIm):           #Ciclo para normalizar ui
        uin[:,m] = ui[:,m]/(math.sqrt(np.sum((ui[:,m])**2))) #Vector/Norma
        #norma = Normalizer().fit(ui.T)   #normalización
        #uin = norma.transform(ui.T).T    #
    
    wi = np.zeros((numIm,numIm))     #Matriz de almacenamiento
    uint = uin.T                     #ui transpuesta
    #plt.figure()
    for n in range(numIm):        #Ciclo para obtener los pesos de cada imagen-
        for p in range(numIm):    #-que está almacenada
            wi[n,p] = uint[p,:] @ A[:,n] #Pesos
            #plt.plot(wi[n,:])
            
    return(uin, psi, k98, wi)
