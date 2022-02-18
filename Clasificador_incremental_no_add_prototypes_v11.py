# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 19:58:25 2021

@author: edgar
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg  as la


def clasificador_incremental(prototypes, sample, label, K):
    
    """
    FUNCION PRINCIPAL
    Recibe los prototipos, una muestra y la etiqueta de la muestra. Cambia
    la posición de los prototipos conforme recibe nuevas muestras.

    Parameters
    ----------
    prototypes : Prototipos.
    sample : Muestra.
    label : Etiqueta de la muestra.

    Returns
    -------
    prototypes : Prototipos actualizados.
    n : Vecinos cercanos de su clase.
    steps : Pasos totales.
    """
    # Revisado
    # La lista tiene ese orden [prototipo][etiqueta prototipo][mem_sampl]
                             # [mem_eti][cls][prom_cls][etq_frontera]
    Feat = 0
    Clas = 1
    Samp = 2
    clss = 4

    # Tamaño maximo de muestras de la memoria de cada prototipo
    maxMemSize = 160
    # Tamaño maximo de fuerza de clasificacion de cada prototipo
    maxMemSizeCls = 40
    # Inicializo numero de pasos
    steps = 0

    # Armar Matriz de prototipos
    Features = listofarray2array(prototypes, Feat) # Convierte lista a matriz
    Class = list2array(prototypes, Clas)            # Convierte lista a array
    
    # Dimensiones de los datos
    data_dim,data_size = Features.shape
    
    # Aplicamos KNN (revisamos si es bien clasificada)
    # Retorna estiqueta estimada, vecinos clase 0, vecinos clase 1, fuerza de clas
    estimated_label,n0,n1,CLS = KNN(sample, Features, Class, K)
    
    # Ptototipos cercanos dependiendo de la etiqueta de la muestra de entrada
    if label == 0:
        # Indices prototipos cercanos clase 0
        n = n0
    else: 
        # Indices prototipos cercanos clase 1
        n = n1
        
    # Si la etiqueta es bien clasificada
    if estimated_label == label:
        # Almacenamos muestra en el prototipo mas cercano
        prototypes = store_sample(prototypes, sample, label, int(n[0,0]), maxMemSize)
        
        feature_back = prototypes[int(n[0,0])][Feat]
        class_back   = prototypes[int(n[0,0])][Clas]
        
        # MLE Anterior
        mleAnt = mem_label_estimation_NN(prototypes,K,prototypes[int(n[0,0])][Feat][:,0].reshape(data_dim,1),10000000)
        
        # Promedio de las posiciones de los prototipos
        tempo = np.concatenate((prototypes[int(n[0,0])][Samp],prototypes[int(n[0,0])][Feat]),axis=1)
        # prototypes[int(n[0,0])][Feat] = np.mean(prototypes[int(n[0,0])][Samp],axis=1).reshape(data_dim,1)  
        prototypes[int(n[0,0])][Feat] = np.mean(tempo,axis=1).reshape(data_dim,1) 
        
        # MLE Posterior
        mlePost = mem_label_estimation_NN(prototypes,K,prototypes[int(n[0,0])][Feat][:,0].reshape(data_dim,1),mleAnt)

        if mleAnt < mlePost:
            # Recupero respaldo
            prototypes[int(n[0,0])][Feat] = feature_back 
            prototypes[int(n[0,0])][Clas] = class_back 
            
        # Se añade CLS a la memoria CLS
        prototypes[int(n[0,0])][clss].append(CLS)
        
        # Si el tamaño es mayor al tamaño maximo definido con maxMemSize
        if len(prototypes[int(n[0,0])][clss]) > maxMemSizeCls:
            # se descarta el mas viejo
            prototypes[int(n[0,0])][clss].pop(0) 
        
        # Calcular promedio fuerza de clasificacion (valor entre 0 y 1)    
        promedio = np.mean(prototypes[int(n[0,0])][clss])                               
        # promedio = np.sum(prototypes[int(n[0,0])][clss])/len(prototypes[int(n[0,0])][clss])
        # Si la fuerza de clasificacion es mayor a 0.9
        if len(prototypes[int(n[0,0])][clss]) == maxMemSizeCls and promedio > 0.995:
            # Se elimina el prototipo porque no es un prototipo frontera
            # del prototypes[int(n[0,0])]
            pass
            # if prototypes[int(n[0,0])][Feat] not in prot_centrales.any():
            #     prot_centrales.append(prototypes[int(n[0,0])][Feat])

    else:
        # Aproximar prototipos cercanos de su clase
        prototypes,steps = approaching_prototype(prototypes, n, sample, label, maxMemSize, K)
  
    return (prototypes, n, steps)

def approaching_prototype(prototypes,n,sample,label,Jmax,KK):
    """
    Aproxima los n prototipos más cercanos a la muestra que está mal estimada.
    Parameters
    ----------
    prototypes : Prototipos
    n : Prototipos más crcanos a la muestra
    sample : muestras
    label : etiqueta de la muestra
    Jmax : Tamaño máximo en memoria

    Returns
    -------
    prototypes : Prototipos actualizados
    steps : Pasos avanzados

    """
    # Revisada parcialmente ------------
    # Pasos que tiene permitido avanzar el algoritmo
    pasosPerm = 25
    # Prototipos cercanos que hay
    K = n.shape[0]
    # Indices para acceder a posicion y clase en las listas de prototipos
    Feat = 0
    Clas = 1

    # Tamaño de la muestra
    data_dim,data_size = prototypes[int(n[0,0])][Feat].shape

    # Creo la memoria de respaldos
    backup_class = []
    backup_features = []
    
    # Respaldo de prototipos
    for k in range(K):
        k = int(k)
        # Respaldo de prototipos cercanos
        backup_features.append(prototypes[int(n[k,0])][Feat][:,0].reshape(data_dim,1))
        backup_class.append(prototypes[int(n[k,0])][Clas])
        # plt.plot(backup_features[k][0],backup_features[k][1],'go')
    
    # Habilitacion de la aproximacion 
    approaching = True
    # Incializamos valor de pasos
    steps = 0
    # Valor esperado de fuerza de clasificacion
    CLS0 = 1
    # Se incia con la muestra 0
    k = 0
    # Numero de pasos para dividir la distancia del prototipo a la muestra
    div = 30
    # 
    deltas_p_v = np.zeros((1,K))
    
        
    while approaching: 
        # Respaldo de posiscion en cada paso
        f_back = prototypes[int(n[k,0])][Feat]
        l_back = prototypes[int(n[k,0])][Clas]
        # Contador
        deltas_p_v[0,k] = deltas_p_v[0,k] + 1
        # Solo se toma el valor de delta la primera vez para que no sea variable
        if deltas_p_v[0,k] == 1:
            # Tamaño del paso
            delta = ((sample) - prototypes[int(n[k,0])][Feat][:,0].reshape(data_dim,1))/div
        

        # MLE antes de hacer la aproximacion
        mle_ant = mem_label_estimation_NN(prototypes,KK,prototypes[int(n[k,0])][Feat][:,0].reshape(data_dim,1),10000000)
        # Aproximacion de prototipos
        prototypes[int(n[k,0])][Feat] = prototypes[int(n[k,0])][Feat][:,0].reshape(data_dim,1) + delta
        
        # MLE despues de aproximacion (Revisar si las memorias se desclasificaron)
        MLE = mem_label_estimation_NN(prototypes,KK,prototypes[int(n[k,0])][Feat][:,0].reshape(data_dim,1),mle_ant)

        # Listas a matrices numpy para usar KNN 
        Features = listofarray2array(prototypes, Feat )
        Class = list2array(prototypes, Clas)

        # KNN para obtener la etiqueta estimada
        estimated_label,n0,n1,CLS = KNN(sample, Features, Class, K)
    
        # Comparacion
        # Si etiqueta es igual a la clase y ninguna memoria se desclasifico
        if estimated_label == label and MLE <= mle_ant:
            #  Alamcenar muestra en memoria de prototipo mas cercano
            # prototypes = store_sample(prototypes, sample, label, int(n[k,0]), Jmax)
            prototypes = store_sample(prototypes, sample, label, int(n[0,0]), Jmax)
            # Redistribuir las muestras
            prototypes = mem_samples_redistribution(prototypes, Jmax,KK)
            # Detengo la aproximacion
            approaching = False 
        
        # Si las etiqueta aun no converge y ninguna muestra se desclasifico 
        elif estimated_label != label and MLE <= mle_ant:
            # Se continua aproximacion
            approaching = True
            # Se aumenta el numero de pasos 
            steps = steps + 1 
            # Si la fuerza de clasificacion mejoro(sige mejorando la aproximacion)
            # y aun no realizamos todos los pasos
            if CLS < CLS0 and steps < pasosPerm:
                # guardo fuerza de la clasificacion
                CLS0 = CLS
                # Y sigo aproximando
                approaching = True
            # Si la fuerza de clasifcacion empeoro
            else:
                # Aproximar siguiente prototipo
                k = k + 1
                # Reincio el contador de pasos
                steps = 0
                # Sigo aproximando
                approaching = True
                #  si es el ultimo prototipo 
                if k >= K :
                    # Detengo aproximacion
                    approaching = False
                    # Para todos los prototipos cercanos
                    # for k in range(K):
                    #     k = int(k)
                    #     # Restaurar sus valores originales
                    #     prototypes[int(n[k,0])][Feat] = backup_features[k]
                    #     prototypes[int(n[k,0])][Clas] = backup_class[k]
    
                    # Agrego la muestra como nuevo prototipo
                    # prototypes = add_sample(prototypes,sample,label)
                    # print("intento")
                    # prototypes = addSampleIfnW(prototypes,protOrig,sample,label,n,KK)
                    # prototypes = mem_samples_redistribution(prototypes, Jmax,KK)
                
        # Si las memorias se malclasificaron
        elif MLE > mle_ant:
            # Retrocede el prototipo un paso 
            prototypes[int(n[k,0])][Feat] = f_back 
            prototypes[int(n[k,0])][Clas] = l_back
            # y se procede con el siguiente prototipo 
            k = k + 1
            # Reinicio los pasos
            steps = 0
            # Continuo aproximacion
            approaching = True
            #  si es el ultimo prototipo
            if k >= K :
                # Detener aproximacion   
                approaching = False
                # # Para todos los prototipos cercanos
                # for k in range(K):
                #     k = int(k)
                #     # Restauro sus valores originales
                #     prototypes[int(n[k,0])][Feat] = backup_features[k]
                #     prototypes[int(n[k,0])][Clas] = backup_class[k]
                # Agregar la muestra como nuevo prototipo
                # prototypes = add_sample(prototypes,sample,label)
                # print("intento")
                # prototypes = addSampleIfnW(prototypes,protOrig,sample,label,n,KK)
                # prototypes = mem_samples_redistribution(prototypes, Jmax,KK)
              
    return (prototypes,steps)

def mem_label_estimation_NN(prototypes,K,ref_sample,mle_ant):
    """
    Revisa la clasificación de las memorias por medio de kNN contabiliza las
    estimaciones incorrectas.
    ---------Funcion alternativa a mem_label_estimation-----''''''

    Parameters
    ----------
    prototypes : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    ref_sample : TYPE
        DESCRIPTION.
    mle_ant : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Revisado
    # Convierto en matriz para poder usar con KNN
    Features = listofarray2array(prototypes, 0)   # Convertimos a matriz
    Class = list2array(prototypes, 1)             # Convertimos a vector
    # Reviso todos los prototipos
    mle_cont = 0
    N = Class.shape[1]         # Número de prototipos
    d = np.zeros((1,N))        # Vector para almacenar distancias

    for i in range(N):
        d[0,i] = la.norm(ref_sample - Features[:,i])
    indices = np.argsort(d)
    
    k = 30  # Reviso los k mas cercanos máximo para evitar gasto computacional
    if N < k:
        k = N
    # k = N
    
    for n in range(k):      # Para cada prototipo
        m = indices[0,n]    # Prorotipo a revisar
        n_m = prototypes[m][3].shape[1]
        # Revisamos cada muestra almacenada en las memorias de prototipos
        for j in range(n_m): 
            estimated,na,nb,ClS = KNN(prototypes[m][2][:,j].reshape(prototypes[m][2].shape[0],1), Features, Class, K)
            # Si la etiqueta es incorrecta la contabilizamos
            if estimated != prototypes[m][3][0,j]:
                mle_cont += 1
                # Si la nueva mle es peor que la anterior salimos del ciclo
                if mle_cont > mle_ant:
                    return(mle_cont)

    return (mle_cont)   

            
def mem_samples_redistribution(prototypes, max_memory_size, K):
    """
    Redistribuye las muestras en memoria de los prototipos a sus vecinos más 
    cercanos.

    Parameters
    ----------
    prototypes : Prototipos.
    max_memory_size : Tamaño máximo de la memoria de prototipos.

    Returns
    -------
    prototypes : Prototipos con memorias actualizadas.

    """
    # Revisada
    # Valores para facilitar la indexacion.
    Feat = 0
    Clas = 1
    Samp = 2
    Labe = 3 

    # Lista a matriz de prototipos 
    Features = listofarray2array(prototypes,Feat)
    Class = list2array(prototypes,Clas)
    # Dimensiones del set de datos para poder ajustar el tamaño
    data_dim,data_size = Features.shape
     
    # Se revisa cada prototipo i es el indice del prototypo
    for i in range(len(prototypes)):   # Para cada prototipo
        # j es el indice de la memoria
        [r,J] = prototypes[i][3].shape # Tamaño de memoria de cada prototipo
        
        if J >= 1:                     # Si hay muestras almacenadas en prototipo
            j = 0                       # Comenzamos con la primera muestra en mem
            while j < J:               # Para cada muestra
                # Se aplica kNN
                # n0,n1 = KNN_estimation(prototypes[i][Samp][:,j].reshape(data_dim,1), Features, Class, K)
                el,n0,n1,fclas = KNN(prototypes[i][Samp][:,j].reshape(data_dim,1), Features, Class, K)
                # Revisamos a que clase pertenece y tomamos vecino mas cercano de la misma clase
                if prototypes[i][Labe][0,0] == 1:
                    n = int(n1[0]) 
                else: 
                    n = int(n0[0])
                    
            # Si el prototipo no esta bien ubicado (la muestra esta mas cerca de otro prototipo)
                if i != n: 
                #  Almacenamos muestra en su mas prototipo cercano n
                    prototypes = store_sample(prototypes, 
                                             prototypes[i][Samp][:,j].reshape(data_dim,1),
                                             prototypes[i][Labe][0,j].reshape(1,1),
                                             n,
                                             max_memory_size)
                    # Eliminamos muestra en prototipo actual
                    prototypes[i][Samp] = np.delete(prototypes[i][Samp],j,axis=1)
                    prototypes[i][Labe] = np.delete(prototypes[i][Labe],j,axis=1)
                    # Decrementamos J en cada ciclo (muestra revisada)
                    J = J - 1
                # Incrementamos indice de muestra
                j = j + 1
    return (prototypes)

def store_sample(prototypes, sample, label, prototype_index,max_memory_size):
    """
    Guarda la muestra (sample, label) en la memoria del prototipo 
    prototipe_index, si se sobrepasa el tamaño máximo de memoria, se elimina a
    la muestra almacenada más vieja.

    Parameters
    ----------
    prototypes : Prototipos.
    sample : Muestra.
    label : Etiqueta.
    prototype_index : Indice de prototipo a almacenar muestra.
    max_memory_size : Tamaño máximo de la memoria.

    Returns
    -------
    prototypes : Prototipos actualizados.

    """
    # Revisado
    # Se añade muestra al protitipo indicado 
    prototypes[prototype_index][2] = np.append(prototypes[prototype_index][2],sample,axis=1)
    prototypes[prototype_index][3] = np.append(prototypes[prototype_index][3],label,axis=1)

    # Se verifica que no tenga mas muestras en la memoria que las permitidas
    if  prototypes[prototype_index][2].shape[1] > max_memory_size:
        # En caso de que sea mayor se elimina la primera (mas antigua)
        prototypes[prototype_index][2] = np.delete(prototypes[prototype_index][2],0,axis=1)
        prototypes[prototype_index][3] = np.delete(prototypes[prototype_index][3],0,axis=1)
        
    return (prototypes)
    

def KNN(vect, prototypes, labels, k):
    ##---SEMI REVISADA------
    """
    KNN(sample, Features, Class, K)
    Clasificador KNN
    Clasificador knn que retorna los indices de sus vecinos mas cercanos por
    clase, la etiqueta estimada y la fuerza de clasificación. El vector y los
    prototipos se introducen en forma de vector columna. Se utiliza ponderado
    gaussiano.
    Parameters
    ----------
    vect : Muestra a clasificar.
    prototypes : prototipos.
    labels : etiquetas de los prototipos.
    k : k vecinos (número de vecinos mas cercanos deseados).

    Returns
    ----------
    label_estimated : etiqueta estimada.
    n1 : vecinos mas cercanos de la clase 1.
    n2 : vecinos mas cercanos de la clase 2.
    CLS : fuerza de clasificación.
    """
    
    label_estimated  = []
    n1 = []
    n2 = []
    CLS = []

    # Calculo la distancia euclidiana
    sub = vect.reshape(prototypes.shape[0],1) - prototypes
    sub = np.power(sub,2)
    sub = sub.sum(axis=0)
    sub = np.sqrt(sub) 
    
    # Vector de indices que ordena las distancias de menor a mayor
    k_min = np.argsort(sub)
    
    # Valor sigma
    sigma = (sub[k_min[0]]/2)
    # if  len(k_min) < 2*k:
    #     sigma = (sub[k_min[-1]])
    # else:
    #     sigma = (sub[k_min[(2*k)-1]])
    
    # Creo dos matrizes del tamaño de K,2
    class0 = np.zeros([k,2])
    class1 = np.zeros([k,2])
    # Creo dos variables para contar
    class0_count = 0
    class1_count = 0
    
    # Se revisan todos lo indices
    for i in k_min:
        # si pertenece a la clase 1
        if labels[:,i]:
            # si la clase aun es menor que K 
            if class1_count < k:
                # Se guarda el indice de la muestra con su distancia
                class1[class1_count,0] = i
                class1[class1_count,1] = sub[i]
                 
                class1_count = class1_count + 1
        # Si pertenece a la clase 0
        else:
            # si la clase aun es menor que K
            if class0_count < k:
                # Se guarda el indice de la muestra con su distancia
                class0[class0_count,0] = i
                class0[class0_count,1] = sub[i]   
                # Incremento de contador
                class0_count = class0_count + 1
                
                
    # Condicion para evitar división entre 0            
    if sigma == 0.0:
        # Asignamos etiqueta del mas cercano 
        label_estimated = labels[:,k_min[0]]
        
        # Retornamos vecinos cercanos de clase, etiqueta y fuerza de votacion
        if label_estimated:
            n1 = np.array([])
            n2 = np.asarray(class1[0,0])
            n2 = n2.reshape(1,1)
        else:
            n1 = np.asarray(class0[0,0])
            n1 = n1.reshape(1,1)
            n2 = np.array([])
            
        CLS = 1
        label_estimated = np.asarray(label_estimated)
        # Convertimos en arrays
        CLS = np.asarray(CLS) 
        return(label_estimated,n1,n2,CLS)
        
    else  : 
        # Tomamos los vecinos mas cercanos(que existan) si no hay k vecinos
        if class1_count < k:
            class1 = class1[:class1_count,:] 
        
        if class0_count < k:
            class0 = class0[:class0_count,:]
            
        # Si hay valores enclass0
        if class0.shape[0] > 0:
            # Votacion ponderada para la clase 0
            w0 = np.exp(-(np.power(class0[:,1],2))/(2*np.power(sigma,2)))
            # La votacion es la suma de los pesos de la clase 0 
            V0 = sum(w0)
        # Si no hay elementos en la clase 0
        else:
            # La votacion es 0
            V0 = 0
        
        # Si hay valores en class1
        if class1.shape[0] > 0:
            # Votación ponderada para la clase 0
            w1 = np.exp(-(np.power(class1[:,1],2))/(2*np.power(sigma,2)))
            # La votacion es la suma de los pesos de la clase 1
            V1 = sum(w1)
        # Si no hay elemntos en la clase  0 
        else:
            # La votacion es  0 
            V1 = 0
        
        # Si la votacion de la clase 0 es mayor o igual a la de la clase 1
        if V0 >= V1:
            # La etiqueta estimada es 0
            label_estimated = 0
            # Si V0 es 0
            if not V0:
                # La fuerza de la clasificacion es 1
                CLS = 1
            # En caso contrario la fuerza se calcula con la formua
            else:
                CLS = 1 - (V1/V0)
        # Si la votafcion de la clase 1 es mayor
        else:
            # La etiqueta estimada es 1
            label_estimated = 1
            # Si la votacion 1 pesa 0
            if not V1:
                # La fuerza de la clasificacion es de 1
                CLS = 1
            # Si la votacion 1 tiene valor, la fuerza se calcula con la formula
            else:
                CLS = 1 - (V0/V1)

        label_estimated = np.asarray(label_estimated)

        n1 = np.asarray(class0[:,0])
        n1 = n1.reshape(len(n1),1)
        n2 = np.asarray(class1[:,0])
        n2 = n2.reshape(len(n2),1)
        CLS = np.asarray(CLS)  

    return (label_estimated,n1,n2,CLS)


def KNNvot(vect, prototypes, labels, k):
    ##---SEMI REVISADA------
    """
    KNN(sample, Features, Class, K)
    Clasificador KNN
    Clasificador knn que retorna los indices de sus vecinos mas cercanos por
    clase, la etiqueta estimada y la fuerza de clasificación. El vector y los
    prototipos se introducen en forma de vector columna. Se utiliza ponderado
    gaussiano.
    Parameters
    ----------
    vect : Muestra a clasificar.
    prototypes : prototipos.
    labels : etiquetas de los prototipos.
    k : k vecinos (número de vecinos mas cercanos deseados).

    Returns
    ----------
    label_estimated : etiqueta estimada.
    n1 : vecinos mas cercanos de la clase 1.
    n2 : vecinos mas cercanos de la clase 2.
    CLS : fuerza de clasificación.
    """
    
    label_estimated  = []
    n1 = []
    n2 = []
    CLS = []

    # Calculo la distancia euclidiana
    sub = vect.reshape(prototypes.shape[0],1) - prototypes
    sub = np.power(sub,2)
    sub = sub.sum(axis=0)
    sub = np.sqrt(sub) 
    
    # Vector de indices que ordena las distancias de menor a mayor
    k_min = np.argsort(sub)
    
    # Valor sigma
    sigma = (sub[k_min[0]]/2)
    # if  len(k_min) < 2*k:
    #     sigma = (sub[k_min[-1]])
    # else:
    #     sigma = (sub[k_min[(2*k)-1]])
    
    # Creo dos matrizes del tamaño de K,2
    class0 = np.zeros([k,2])
    class1 = np.zeros([k,2])
    # Creo dos variables para contar
    class0_count = 0
    class1_count = 0
    
    # Se revisan todos lo indices
    for i in k_min:
        # si pertenece a la clase 1
        if labels[:,i]:
            # si la clase aun es menor que K 
            if class1_count < k:
                # Se guarda el indice de la muestra con su distancia
                class1[class1_count,0] = i
                class1[class1_count,1] = sub[i]
                 
                class1_count = class1_count + 1
        # Si pertenece a la clase 0
        else:
            # si la clase aun es menor que K
            if class0_count < k:
                # Se guarda el indice de la muestra con su distancia
                class0[class0_count,0] = i
                class0[class0_count,1] = sub[i]   
                # Incremento de contador
                class0_count = class0_count + 1
                
                
    # Condicion para evitar división entre 0            
    if sigma == 0.0:
        # Asignamos etiqueta del mas cercano 
        label_estimated = labels[:,k_min[0]]
        
        # Retornamos vecinos cercanos de clase, etiqueta y fuerza de votacion
        if label_estimated:
            n1 = np.array([])
            n2 = np.asarray(class1[0,0])
            n2 = n2.reshape(1,1)
        else:
            n1 = np.asarray(class0[0,0])
            n1 = n1.reshape(1,1)
            n2 = np.array([])
            
        CLS = 1
        label_estimated = np.asarray(label_estimated)
        # Convertimos en arrays
        CLS = np.asarray(CLS) 
        return(label_estimated,n1,n2,CLS)
        
    else  : 
        # Tomamos los vecinos mas cercanos(que existan) si no hay k vecinos
        if class1_count < k:
            class1 = class1[:class1_count,:] 
        
        if class0_count < k:
            class0 = class0[:class0_count,:]
            
        # Si hay valores enclass0
        if class0.shape[0] > 0:
            # Votacion ponderada para la clase 0
            w0 = np.exp(-(np.power(class0[:,1],2))/(2*np.power(sigma,2)))
            # La votacion es la suma de los pesos de la clase 0 
            V0 = sum(w0)
        # Si no hay elementos en la clase 0
        else:
            # La votacion es 0
            V0 = 0
        
        # Si hay valores en class1
        if class1.shape[0] > 0:
            # Votación ponderada para la clase 0
            w1 = np.exp(-(np.power(class1[:,1],2))/(2*np.power(sigma,2)))
            # La votacion es la suma de los pesos de la clase 1
            V1 = sum(w1)
        # Si no hay elemntos en la clase  0 
        else:
            # La votacion es  0 
            V1 = 0
        
        # Si la votacion de la clase 0 es mayor o igual a la de la clase 1
        if V0 >= V1:
            # La etiqueta estimada es 0
            label_estimated = 0
            # Si V0 es 0
            if not V0:
                # La fuerza de la clasificacion es 1
                CLS = 1
            # En caso contrario la fuerza se calcula con la formua
            else:
                CLS = 1 - (V1/V0)
        # Si la votafcion de la clase 1 es mayor
        else:
            # La etiqueta estimada es 1
            label_estimated = 1
            # Si la votacion 1 pesa 0
            if not V1:
                # La fuerza de la clasificacion es de 1
                CLS = 1
            # Si la votacion 1 tiene valor, la fuerza se calcula con la formula
            else:
                CLS = 1 - (V0/V1)

        label_estimated = np.asarray(label_estimated)

        n1 = np.asarray(class0[:,0])
        n1 = n1.reshape(len(n1),1)
        n2 = np.asarray(class1[:,0])
        n2 = n2.reshape(len(n2),1)
        CLS = np.asarray(CLS)  

    return (label_estimated,n1,n2,CLS,V0,V1)



def KNN_estimation(vect, prototypes, labels, k):
    ##---SEMI REVISADA------

    label_estimated  = []
    n1 = []
    n2 = []
    CLS = []

    # Calculo la distancia euclidiana
    sub = vect.reshape(prototypes.shape[0],1) - prototypes
    sub = np.power(sub,2)
    sub = sub.sum(axis=0)
    sub = np.sqrt(sub) 
    
    # Vector de indices que ordena las distancias de menor a mayor
    k_min = np.argsort(sub)
    
    # Valor sigma
    sigma = (sub[k_min[0]]/2)
    # if  len(k_min) < 2*k:
    #     sigma = (sub[k_min[-1]])
    # else:
    #     sigma = (sub[k_min[(2*k)-1]])
    
    # Creo dos matrizes del tamaño de K,2
    class0 = np.zeros([k,2])
    class1 = np.zeros([k,2])
    # Creo dos variables para contar
    class0_count = 0
    class1_count = 0
    
    # Se revisan todos lo indices
    for i in k_min:
        # si pertenece a la clase 1
        if labels[:,i]:
            # si la clase aun es menor que K 
            if class1_count < k:
                # Se guarda el indice de la muestra con su distancia
                class1[class1_count,0] = i
                class1[class1_count,1] = sub[i]
                 
                class1_count = class1_count + 1
        # Si pertenece a la clase 0
        else:
            # si la clase aun es menor que K
            if class0_count < k:
                # Se guarda el indice de la muestra con su distancia
                class0[class0_count,0] = i
                class0[class0_count,1] = sub[i]   
                # Incremento de contador
                class0_count = class0_count + 1
                
                
    # Condicion para evitar división entre 0            
    if sigma == 0.0:
        # Retornamos vecinos cercanos de clase, etiqueta y fuerza de votacion
        if label_estimated:
            n1 = np.array([])
            n2 = np.asarray(class1[0,0])
            n2 = n2.reshape(1,1)
        else:
            n1 = np.asarray(class0[0,0])
            n1 = n1.reshape(1,1)
            n2 = np.array([])
            
        CLS = 1
        label_estimated = np.asarray(label_estimated)
        # Convertimos en arrays
        CLS = np.asarray(CLS) 
        return(n1,n2)
        
    else  : 
        # Tomamos los vecinos mas cercanos(que existan) si no hay k vecinos
        if class1_count < k:
            class1 = class1[:class1_count,:] 
        
        if class0_count < k:
            class0 = class0[:class0_count,:]
            

        n1 = np.asarray(class0[:,0])
        n1 = n1.reshape(len(n1),1)
        n2 = np.asarray(class1[:,0])
        n2 = n2.reshape(len(n2),1)

    return (n1,n2)


def add_sample(prototypes,sample,label):
    """
    Se almacena la muestra (sample) como un nuevo prototipo.

    Parameters
    ----------
    prototypes : Prototipos.
    sample : Muestra a convertir en prototipo.
    label : Etiqueta de la muestra.

    Returns
    -------
    prototypes : Prototipos actualizados.
    """
    # Indice de posicion en lista de prototipos
    Feat = 0
    # Dimension de los datos
    data_dim,data_size = prototypes[0][Feat].shape
    # Creamos prototipos
    Prototype = []
    CLS = []
    # Agregamos muestra
    Prototype.append(sample.reshape(data_dim,1))
    # Agragamos etiqueta
    Prototype.append(int(label))
    # Agregamos su posicion en su memoria
    Prototype.append(sample.reshape(data_dim,1))
    # Agregamos etiqueta en su memoria
    Prototype.append(np.array(label.reshape(1,1)))
    # Agregamos fuerza de clasificacion
    Prototype.append(CLS)
    # Agregamos el prototipo a la lista de prototipos
    prototypes.append(Prototype)
    
    return(prototypes)

def add_sampleIni(prototypes, sample, label):
    """
    Se almacena la muestra (sample) como un nuevo prototipo.

    Parameters
    ----------
    prototypes : Prototipos.
    sample : Muestra a convertir en prototipo.
    label : Etiqueta de la muestra.

    Returns
    -------
    prototypes : Prototipos actualizados.
    """

    # Creamos prototipos
    Prototype=[]
    CLS= []
    # Agregamos muestra
    Prototype.append(sample.reshape(-1,1))
    # Agragamos etiqueta
    Prototype.append(int(label))
    # Agregamos su posicion en su memoria
    Prototype.append(sample.reshape(-1,1))
    # Agregamos etiqueta en su memoria
    Prototype.append(np.array(label.reshape(1,1)))
    # Agregamos fuerza de clasificacion
    Prototype.append(CLS)
    # Agregamos el prototipo a la lista de prototipos
    prototypes.append(Prototype)
    
    return (prototypes)



#%% Funciones para realizar Graficas 
def graf_sets(in_data, in_eti,titulo,x_lab,y_lab):
    """ Genera una grafica donde se ve el valor del set de datos 
    y sus etiquetas ( valido para 2 dimensiones y solo 2 tipos de datos).
    
    Entradas:
        
                in_data: Datos de entrada.
                
                in_eti:  Etiquetas de los datos  (a1 y 0).
                
                titulo:  Titulo de la grafica.
                
                x_lab:   Titulo del eje X.
                
                y_lab:   Titulo del eje Y.
                
    Salidas: 
        
                Muestra la grafica, mas no devuelve ningun valor.
    """
    # cambiar muestras de numeros a rojo y azul
    lab = []
    # print ("test",in_eti.shape[1])
    for i in range(in_eti.shape[1]):
        # print(i)
        if in_eti[0,i]:
            lab.append( "blue")
        else:
            lab.append( "red")
            
    # Graficar datos z
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(titulo)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    # ax.set_aspect('equal', adjustable='box')
    plt.scatter(in_data[0,:],in_data[1,:],color=lab,s=60)
    # plt.plot(sample[0],sample[1],'co')
    plt.tight_layout()
    plt.show()

def graf_pross(prototypes,x,y, title):
    Features = listofarray2array(prototypes, x )
    Class = list2array(prototypes, y)
    graf_sets(Features,Class,title,"X","Y")



# %%Funciones par convertir a matris las listas 
def listofarray2array(input_list,index_a):
    """ Funcion que extrae de una lista de listas, un array comun de las listas
    el array extraido es el del index_a.
    
    Entradas:
        
                input_list:         Lista de entrada.
                
                index_a:            Indice del arreglo a extraer.

    Salidas: 
        
                extracted_list:     Se devuelve la lista extraida en arreglo de numpy.
    """
    # DEB = False
    start = True
    # Para todos los indices en la lista
    for index in range(len(input_list)):
        # Si es el primer valor
        if start:
            # if DEB: print("entrada",input_list[index][index_a])
            # if DEB: print("entrada:size",input_list[index][index_a].shape)
            # Se extrae el sub arreglo del indice a
            
            extracted_list = input_list[index][index_a]
            start=False
        else:
            # if DEB: print("entrada",input_list[index][index_a])
            # Se extrae el sub arreglo del indice a y se anexa a la lista
            extracted_list = np.append(extracted_list,input_list[index][index_a],axis=1)
    # Se regresa la lista
    return (extracted_list)

def list2array(input_list,index_a):
    """ Funcion que extrae de una lista de listas, una lista comun de las listas
    la lista extraida es de index_a.
    
    Entradas:
        
                input_list:         Lista de entrada.
                
                index_a:            Indice de la lista a extraer.

    Salidas: 
        
                extracted_list:     Se devuelve la lista extraida en arreglo de numpy.
    """

    # DEB= False
    # Se crea una listqa vacia
    extracted_list=[]
    # Para todas las listas en la lista 
    for index in range(len(input_list)):
        # if DEB: print("entrada",input_list[index][index_a])
        # Se hace append de todas las listas que conciden
        extracted_list.append(input_list[index][index_a])
    # Se convierte en arreglo de numpy
    # print("append",extracted_list)
    extracted_list=np.array(extracted_list)
    # print("array",extracted_list)
    # Se verifica la estructura del arreglo
    extracted_list=extracted_list.reshape(1,extracted_list.shape[0])
    # print("reshape",extracted_list)
    # Se regresa en forma de arreglo
    return (extracted_list)


# Función guardar datos
def guardar_datos(archivo,datos):
    with open(archivo, "wb") as f:
        pickle.dump(datos, f)
        
# Función knn test        
def knn_test(Prototypes,T_tst,S_tst, L_tst, K):
    # 0 NORMAL, 1 NEUMONIA
    data_dim= S_tst.shape[0]           # Obtenemos la dimensión de características
    Feat = 0
    Clas = 1
    Prototypes_tst = listofarray2array(Prototypes, Feat ) # Prototipos
    Prototypes_labels_tst = list2array(Prototypes, Clas)  # Etiquetas
    contador_aciertos = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # accuracy = 0
    recognition_rate = 0
    for j in range(T_tst):
        estimated_label_tst,n1_10,n0_10,CLS_10 = KNN(S_tst[:,j].reshape(data_dim,1), Prototypes_tst, Prototypes_labels_tst, K)
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
    return(recognition_rate,TP,TN,FP,FN)

# Normalizacion
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

# Proyeccion
def proy1Tst(vec,eigenUi,psi):
    pesos = np.zeros((eigenUi.shape[1],1))
    imgMnsprom = vec - psi
    pesos[:,0] = eigenUi.T @ imgMnsprom
    
    return(pesos)