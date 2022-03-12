# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 23:02:36 2021

@author: edgar
"""

import time
import pickle    
import funciones               
import numpy as np
from numpy import linalg as la        
import Clasificador_incremental_no_add_prototypes_v11 as cinap 


# Normalización de datos
def normStd(x):
    """
    Obtenemos el maximo y el minimo de cada caracteristica (filas), obtenemos 
    el rango en que varian. Para normalizar dividimos entre el rango y le 
    sumamos el minimo.

    Parameters
    ----------
    x : Datos de entrenamiento (vectores columna)
    -------
    Returns
    ----------

    """
    promedio = np.mean(x,axis=1)
    desvStd = np.std(x,axis=1)
    desvStd = 1/desvStd
    prom = -promedio/desvStd
    datNorm = np.zeros((x.shape))
    for i in range(x.shape[0]):
        datNorm[i,:] = desvStd[i]*x[i,:] + prom[i]
        
    return (datNorm,desvStd,prom)

def normStdTst(x,desvStd,prom):
    """
    Parameters
    ----------
    x : TYPE
        Datos en forma de vector columna.

    Returns
    -------
    datNorm : TYPE
        Datos normalizados.
    """
    datNorm = np.zeros((x.shape))
    for i in range(x.shape[0]):
        datNorm[i,:] = desvStd[i]*x[i,:] + prom[i]
        
    return (datNorm)

def eigenEspacio(vecIm):
    # Promedio
    psi = np.mean(vecIm, axis=1)                    # Vector promedio de imagenes
    # Resta del promedio a cada imagen
    A = funciones.resta_psi(vecIm, psi) 
    # Matriz de covarianza
    matcov = A.T @ A
    # Eigenvectores y eigenvalores
    valpro, vecpro = la.eig(matcov)             # Eigenvectores y eigenvalores 
    valpro = np.real(valpro)                    # Parte real de los valore propios
    vecpro = np.real(vecpro)                    # Parte real de los vectores propios
    # Valores propios ordenados
    ind = valpro.argsort()        # Indices que ordenan de menor a mayor
    indice = ind[::-1]            # Ordenar de los indices mayor a menor np.flip(arr)
    valproord = valpro[indice]    # Eigenvalores ordenados
    # Vectores propios ordenadoss
    vi = vecpro[:,indice]         # Eigenvectores ordenados de mayor a meno
    # Num eigenfaces necesarios para representar el 98 % de la variacion total
    k98 = funciones.k_98(valproord, vecIm.shape[1])    
    # Eigenfaces
    ui = A @ vi                   # Eigenfaces
    # Normalizacion
    uin = funciones.normalizacion(ui, vecIm.shape[1])  # ui normalizado
    # Pesos
    wi = funciones.ponderado(uin, A, vecIm.shape[1])   # Matriz de ponderado
    
    return (uin,wi.T,psi)

# Proyecciones entrenmiento
def proyTst(datos,eigenUi,psi):
    pesos = np.zeros((eigenUi.shape[1],datos.shape[1]))
    for i in range(datos.shape[1]):   
        imgMnsprom = datos[:,i] - psi
        pesos[:,i] = eigenUi.T @ imgMnsprom
    
    return pesos

# %% Numero de imágenes a procesar
numImNorm = 1000
numImNeum = 1000
numImNormTst = 341
numImNeumTst = 345

# %% Rutas de imagenes
# Carpetas de entrenamiento
rutaNormTrain = r'C:\Users\Train\NORMAL'
rutaNeumTrain = r'C:\Users\Train\Viral Pneumonia'
rutaNormTrain.replace('\\', '/')
rutaNeumTrain.replace('\\', '/')
# Carpetas de prueba
rutaNormTest = r'C:\Users\Test\NORMAL'
rutaNeumTest = r'C:\Users\Test\Viral Pneumonia'
rutaNormTest.replace('\\', '/')
rutaNeumTest.replace('\\', '/')

# %% Eigenfaces
# Matriz de imagenes con ambas clases
vecIm0 = funciones.cargar_imag_equHist(rutaNormTrain, numImNorm) # Matriz de imagenes
vecIm1 = funciones.cargar_imag_equHist(rutaNeumTrain, numImNeum) # Matriz de imagenes
vecIm = np.concatenate((vecIm0,vecIm1),axis=1)           # Conjunto total
# Calculo de eigenfaces
uin,wiT,psi = eigenEspacio(vecIm)
# Conjunto de entrenamiento eiquetado
datEntO = wiT
y = np.concatenate((np.zeros((1,numImNorm)),np.ones((1,numImNeum))),1)
datEntO = np.concatenate((datEntO,y),0)

# %% Conjunto de prueba
# Cargamos imagenes vector columna
vecIm0Tst = funciones.cargar_imag_equHist(rutaNormTest, numImNormTst) # Matriz de imagenes
vecIm1Tst = funciones.cargar_imag_equHist(rutaNeumTest, numImNeumTst) # Matriz 
vecImTst = np.concatenate((vecIm0Tst,vecIm1Tst),axis=1)
# Proyeccion de conjunto de entrenamiento
datTstO = proyTst(vecImTst, uin, psi)
# Etiquetado
y = np.concatenate((np.zeros((1,numImNormTst)),np.ones((1,numImNeumTst))),1)
datTstO = np.concatenate((datTstO,y),0)

# %% Normalizamos datos de entrenamiento y prueba
# Normalizamos datos entrenamiento
[datEntNorm,desvStd,prom] = normStd(datEntO[:-1,:])
# Normalizamos datos prueba
datTstNorm = normStdTst(datTstO[:-1,:],desvStd,prom)

# %% Entrenamiento    
# Separamos las clases 
clas0 = datEntNorm[:,:numImNorm] # Cada columna es una muestra
clas1 = datEntNorm[:,numImNorm:] # Cada columna es una muestra

# %% Cálculo de factor de discriminación
nbins = 100
wj = np.zeros((1,numImNorm + numImNeum))
for i in range(numImNorm + numImNeum):
    u0 = np.mean(clas0[i,:])
    u1 = np.mean(clas1[i,:])
    wj[0,i] = (u0-u1)**2 / (np.var(clas0[i,:]) + np.var(clas1[i,:]))

# Indices de factor discriminante mayor a umbral
a = np.array(np.where(wj[0,:]>0.0025)) 
indicesMayores = a.reshape(a.shape[1])

# Seleccionamos características mas discriminantes
# Conjunto de entrenamiento
clas0NormSel = clas0[indicesMayores,:]
clas1NormSel = clas1[indicesMayores,:]
# Conjunto de prueba
datTstNormSel = datTstNorm[indicesMayores,:] 

# Conjunto de entrenamiento y prueba Normalizado y etiquetado
# Conjunto de entrenamiento
datEntNormSel = np.concatenate((clas0NormSel,clas1NormSel),axis=1)
datEntNormSel = np.concatenate((datEntNormSel,datEntO[-1,:].reshape(1,2000)),axis=0)
# Conjunto de prueba
datTstNormSel = np.concatenate((datTstNormSel, datTstO[-1,:].reshape(1,686)),axis=0)

# %% Valores para indexar mas legiblemente las listas 
Feat = 0  # Características
Clas = 1  # Clase
Samp = 2  # Memoria muestras
Labe = 3  # Memoria etiquetas

# %% SEMILLAS
np.random.seed(70) #1

# %% Tomamos prototipos iniciales
Prototypes = []
numProt = 100
ic0 = np.arange(numProt)
np.random.shuffle(ic0)
ic1 = np.arange(numProt)
np.random.shuffle(ic1)
for i in range(numProt):
    cinap.add_sampleIni(Prototypes, clas0NormSel[:,ic0[i]], np.array([0]))
    cinap.add_sampleIni(Prototypes, clas1NormSel[:,ic1[i]], np.array([1]))

# %% # Conjunto de entrenamiento y prueba
T_ent = numImNorm + numImNeum      # Tamaño del conjunto de entrenamiento
indic = np.arange(T_ent)           # Vector de indices
np.random.shuffle(indic)           # Desordenamos vector de indices
datEnt = datEntNormSel[:,indic]    # Desordenamos conjunto de entrenamiento
L_ent = np.zeros((1,T_ent))        # Vector para etiquetas
L_ent[0,:] = datEnt[-1,:]          # Extraemos etiquetas
S_ent = datEnt[:-1,:]              # Retiramos etiquetas de conjunto de ent.
data_dim= S_ent.shape[0]           # Obtenemos la dimensión de características 

T_tst = numImNormTst + numImNeumTst # Tamaño del conjunto de test
L_tst = np.zeros((1,T_tst))         # Vector de etiquetas
L_tst[0,:] = datTstNormSel[-1,:]    # Extraemos etiquetas 
S_tst = datTstNormSel[:-1,:]        # Retiramos etiquetas conjunto de prueba

# %% Entrenamiento
tic=time.time()

#for ki in range(1,16):    
for ki in [1,3,5,7,9,11,13,15]:
    tic = time.time()
    print(ki)
    # Para todas las muestras 
    for i in range(T_ent):
        Prototypes, n, steps = cinap.clasificador_incremental(Prototypes, S_ent[:,i].reshape(data_dim,1), L_ent[:,i].reshape(1,1),ki)
        if i == T_ent-1:
            [recognition_rate,TP,TN,FP,FN] = cinap.knn_test(Prototypes, T_tst, S_tst, L_tst, ki)
    
    # Muestro el tiempo total que requirio el procedimiento
    print(recognition_rate)
    print("Num Prot:",len(Prototypes))
    print('Tiempo: {} min'.format((time.time()-tic)/60))
    print("TP, TN, FP, FN:")
    print(TP, TN, FP, FN)

# %% Utilizando todas las muestras
for ki in range(15):
    tic=time.time()
    # 0 NORMAL, 1 NEUMONIA
    data_dim= S_tst.shape[0]           # Obtenemos la dimensión de características
    contador_aciertos = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    recognition_rate = 0
    for j in range(T_tst):
        estimated_label_tst,n1_10,n0_10,CLS_10 = cinap.KNN(S_tst[:,j].reshape(data_dim,1), S_ent, L_ent, ki+1)
        if estimated_label_tst == L_tst[:,j]:
            if estimated_label_tst == 1:
                TP += 1
            else:
                TN += 1
            contador_aciertos = contador_aciertos + 1
        else:
            if estimated_label_tst == 1:
                FP += 1
            else:
                FN += 1
    recognition_rate = contador_aciertos/T_tst
    # accuracy = (TP+TN)/(TP+TN+FP+FN)
    print(contador_aciertos)
    print("Recognition rate = ", recognition_rate)
    print('Tiempo: {} min'.format((time.time()-tic)/60))

# %% GUARDAR PROTOTIPOS   
# cinap.guardar_datos("prototiposOVOnvpk15.pkl", Prototypes)
# cinap.guardar_datos("uinOVOnvpk15.pkl", uin)
# cinap.guardar_datos("psiOVOnvpk15.pkl", psi)
# cinap.guardar_datos("indicesOVOnvpk15.pkl", indicesMayores)
# cinap.guardar_datos("desvStdOVOnvpk15.pkl", desvStd)
# cinap.guardar_datos("promOVOnvpk15.pkl", prom)
# cinap.guardar_datos("datEntOVOnvpk15.pkl", datEntNormSel)
# cinap.guardar_datos("datTstOVOnvpk15.pkl", datTstNormSel)
