# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:54:41 2018

@author: kingSelta

Basado en las funciones provistas en el curso de redes neuronales de la 
maestría de data mining de la Universidad de Buenos Aires
2018
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def purelin(x):
    return x

def dpurelin(x):
    return np.ones_like(x)
    
def logsig(x):
    return 1 / (1 + np.exp(-x))

def dlogsig(x):
    return logsig(x) * (1 - logsig(x))

def tansig(x):
    return np.tanh(x)
    #return 2 / (1 + np.exp(-2 * x)) - 1

def dtansig(x):
    return 1.0 - np.square(tansig(x))

def gaussian(x, mu, sig): # Gaussian 1D
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def dsoftmax(x):
    s = softmax(x)
    n = s.shape[0]
    jacobian_m = np.zeros((n,n))
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m



#------------------------------------------------------------------------------
# ABRIR IMAGEN
#------------------------------------------------------------------------------
# En Python 2.7 hay que hacer otro import que ofrece imread (quizas scipy.image)
def AbrirImagen(archivo):
    """
    Parámetros:
       archivo: Es un archivo con una imagen en formato bmp con una profundidad
                en bits de 4, a la cual se le reconoceran todos los puntos que
                no sean de color blanco y según la paleta de 256 colores genera
                una etiqueta o clase a cada punto según su color
    
    Devuelve:
           X: Una matriz de n filas (la cantidad de muestras es decir puntos en la 
              imagen) por 3 columnas que contienen la cordenada X e Y de cada punto
              y la clase (color) al que pertenecen.
    
    Ejemplo de uso:
        archivo = 'D:\\Facultad\\Catedras\\UBA\\Teorías\\04_BPN\\5 clases 2.bmp'
        X = AbrirImagen(archivo)
        T = X[:, 2]
        P = X[:, 0:2]
        bpnPlot(P, T, np.array([[0,0]]), [0])
    """
    # HINT: imread es depreciado y removido en SciPy 1.2.0. En ese caso se debe
    # ajustar para usar imageio.imread
    datos = np.array(sp.misc.imread(archivo, mode='P'))   # en mac es mode='I'
    maximo = len(datos)
    X = np.array([], dtype=np.int64).reshape(0,3)
    
    # HINT: El número corresponde al número del color en cuestion en la lista de 256 colores
    # debido a que la imagen se carga en modo P
    colores = np.array([0, 9, 12, 10, 6, 11]) # negro rojo azul verde teal amarillo. 
    for color in colores:
        filas, columnas = np.where(datos == color)
        clase = np.where(colores == color)[0][0] 
        clases = [clase] * len(filas)
        X = np.vstack([X, np.column_stack((columnas+1, maximo-(filas+1), clases))])
    return X



#------------------------------------------------------------------------------
# PERCEPTRON
#------------------------------------------------------------------------------
def perceptronPlot(P, T, W, b):
    """  
    Parámetros:
           P: es una matriz con los datos de los patrones con los cuales
               entrenar el perceptrón. Los ejemplos deben estar en filas.
           T: es un vector con la clase esperada para cada ejemplo. Los
               valores de las clases deben ser 0 (cero) y 1 (uno)
           W: la matriz de pesos W del percpetrón entrenado
           b: valor del bias (W0) del perceptrón entrenado
    
    Ejemplo de uso:
           perceptronPlot(P, T, W, b);

    """
    plt.clf()
    
    #ceros
    x0=[]
    y0=[]
    x1=[]
    y1=[]
    for i in range(len(T)):
        if T[i] == 0:
            x0.append(P[i, 0])
            y0.append(P[i, 1])
        else:
            x1.append(P[i, 0])
            y1.append(P[i, 1])
            
    plt.scatter(x0, y0, marker='+', color='b')            
    plt.scatter(x1, y1, marker='o', color='g')
    
    #ejes
    minimos = np.min(P, axis=0)
    maximos = np.max(P, axis=0)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    plt.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    
    #recta discriminante
    m = W[0,0] / W[0,1] * -1
    n = b / W[0,1] * -1
    x1 = minimos[0]
    y1 = x1 * m + n
    x2 = maximos[0]
    y2 = x2 * m + n
    plt.plot([x1, x2],[y1, y2], color='r')
    
    plt.draw()
    plt.pause(0.00001)   


def perceptronTrain(P, T, alfa, max_itera, dibujar, W_0 = None, b_0 = None):
    """
    Parámetros:
           P: es una matriz con los datos de los patrones con los cuales
               entrenar el perceptrón. Los ejemplos deben estar en filas.
           T: es un vector con la clase esperada para cada ejemplo. Los
               valores de las clases deben ser 0 (cero) y 1 (uno)
           alfa: velocidad de aprendizaje
           max_itera: la cantidad máxima de iteraciones en las cuales se va a
               ejecutar el algoritmo
           dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
               ejemplos y la recta discriminante.
           W_0: (Opcional) Vector de pesos iniciales, debe tener tantos 
                elementos como entradas (atributos) tenga el perceptron.
           b_0: (Opcional) Valor inicial del bias.
    
    Devuelve:
           W: la matriz de pesos W del percpetrón entrenado
           b: valor del bias (W0) del perceptrón entrenado
           ite: número de iteraciones ejecutadas durante el algoritmo. Si
               devuelve el mismo valor que MAX_ITERA es porque no pudo finalizar
               con el entrenamiento
    
    Ejemplo de uso:
           [W, b, ite] = perceptronTrain(P, T, 0.25, 250, True);           
    """    
        
    (cant_patrones, cant_atrib) = P.shape
    
    # Usa los pesos iniciales y bias ingresados, en caso de no ser enviados
    # utiliza valores al azar.
    if W_0 is None:
        W = np.random.rand(1, cant_atrib)
    else:
        W = W_0
    if b_0 is None:
        b = np.random.rand()
    else:
        b = b_0
    

    ite = 0
    otra_vez = True
    
    plt.ion()
    plt.show()
    
    while ((ite <= max_itera) and otra_vez):
        otra_vez = False
        ite = ite + 1
        
        for patr in range(cant_patrones):
            salida = b + W.dot(P[patr, :][np.newaxis].T) 
            if salida >= 0:
                salida = 1
            else:
                salida = 0
      
            factor = alfa * (T[patr] - salida)
            if (factor != 0):
                otra_vez = True
                W = W + factor * P[patr, :][np.newaxis]
                b = b + factor
        
        if dibujar and (cant_atrib == 2):        
            perceptronPlot(P, T, W, b)            

    if dibujar and (cant_atrib == 2):        
        perceptronPlot(P, T, W, b)
        
    return (W, b, ite)


def perceptronPredict(P, W, b, dibujar):
    """
    Parámetros:
           P: es una matriz con los datos de los patrones con los cuales
               entrenar el perceptrón. Los ejemplos deben estar en filas.
           W: la matriz de pesos W del percpetrón entrenado
           b: valor del bias (W0) del perceptrón entrenado
           dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
               ejemplos y la recta discriminante.
    Devuelve:
           T: es un vector con la clase esperada para cada ejemplo. Los
               valores de las clases deben ser 0 (cero) y 1 (uno)
    
    Ejemplo de uso:
           T = perceptronPredict(P, W, 0.25, True);
           
    """    
    
    T = []
    if P.shape[1] != W.shape[1]:
        print("Error! la matríz de pesos debe tener tantas columnas (atributos) \
              como atributos (variables) tenga cada individuo a predecir (columnas en P)") 
    else:
        for individuo in P:
            funNeta = 0
            for (individuoAtrib, peso) in zip(individuo, W[0]):
                funNeta = funNeta + (individuoAtrib * peso)
            funNeta = funNeta + b
            if  funNeta >= 0:
                T.append([1])
            else:
                T.append([0])
        T = np.asarray(T)
                
        # Gráfica si se requierió y la dimensionalidad de los individuos
        # es igual a 2
        if (dibujar and P.shape[1] == 2):
            perceptronPlot(P, T, W, b)
    
        return T

#------------------------------------------------------------------------------
# ADALINE
#------------------------------------------------------------------------------    
def adalinePlot(P, T, W, b, title = ''):
    """     
    Parámetros:
           P: es una matriz con los datos de los patrones con los cuales
               entrenar el perceptrón. Los ejemplos deben estar en filas.
           T: es un vector con la clase esperada para cada ejemplo. Los
               valores de las clases deben ser 0 (cero) y 1 (uno)
           W: la matriz de pesos W del percpetrón entrenado
           b: valor del bias (W0) del perceptrón entrenado
           title: el título que aparecerá en la gráfica
    
    Ejemplo de uso:
           adalinePlot(P, T, W, b, 'Entrenamiento final del Adaline');
    """
    
    plt.clf()
    #ceros
    x=[]
    y=[]
    for i in range(len(T)):
        if T[i] == 0:
            x.append(P[i, 0])
            y.append(P[i, 1])
    plt.scatter(x, y, marker='+', color='b')
    
    #unos
    x=[]
    y=[]
    for i in range(len(T)):
        if T[i] == 1:
            x.append(P[i, 0])
            y.append(P[i, 1])
    plt.scatter(x, y, marker='o', color='g')
    
    #ejes
    minimos = np.min(P, axis=0)
    maximos = np.max(P, axis=0)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    plt.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    
    #recta discriminante
    m = W[0,0] / W[0,1] * -1
    n = b / W[0,1] * -1
    x1 = minimos[0]
    y1 = x1 * m + n
    x2 = maximos[0]
    y2 = x2 * m + n
    plt.plot([x1, x2],[y1, y2], color='r')
    
    plt.title(title)
    
    plt.draw()
    plt.pause(0.00001) 
    
    
    
def adalineTrain(P, T, alfa, max_itera, cota_error, funcion, dibujar):
    """
    Parámetros:
           P: es una matriz con los datos de los patrones con los cuales
               entrenar el perceptrón. Los ejemplos deben estar en filas.
           T: es un vector con la clase esperada para cada ejemplo. Los
               valores de las clases deben ser acordes a la función de transferencia 
               utilizada:
                   - para logsig: 0 (cero) y 1 (uno) 
                   - para tansig: -1(menos uno) y 1 (uno)
           alfa: velocidad de aprendizaje
           max_itera: la cantidad máxima de iteraciones en las cuales se va a
               ejecutar el algoritmo
           cota_error: error promedio mínimo que se espera alcanzar como condición 
               de fin del algoritmo
           funcion: un string con el nombre de la función de transferencia a utilizar
               - 'logsig'
               - 'tansig'
           dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
               ejemplos y la recta discriminante.
    
    Devuelve:
           W: la matriz de pesos W del percpetrón entrenado
           b: valor del bias (W0) del perceptrón entrenado
           ite: número de iteraciones ejecutadas durante el algoritmo. Si
               devuelve el mismo valor que MAX_ITERA es porque no pudo finalizar
               con el entrenamiento
           error_prom: error promedio cometido en la última iteración del algoritmo
    
    Ejemplo de uso:
           [W, b, ite, error_prom] = adalineTrain(P, T, 0.01, 1000, 0.001, 'tansig', True)    
       
    """
    (cant_patrones, cant_atrib) = P.shape    
    W = np.random.rand(1, cant_atrib)
    b = np.random.rand()
    
    T2 = T.copy();
    T2 = np.floor((T2 + 1) / 2)

    ite = 0;
    error_prom = 1
    
    while (ite < max_itera) and (error_prom > cota_error):
        SumaError = 0
        for p in range(cant_patrones): 
           neta = b + W.dot(P[p, :])
           
           salida = eval(funcion + '(neta)')            
           errorPatron = T[p] - salida
           SumaError = SumaError + errorPatron ** 2
           
           derivada = eval('d' + funcion + '(neta)')
           
           grad_b = -2 * errorPatron * derivada;
           grad_W = -2 * errorPatron * derivada * P[p, :]

           b = b - alfa * grad_b;
           W = W - alfa * grad_W;         
        
        error_prom = SumaError / cant_patrones
        ite = ite + 1
        print(ite, error_prom)   
        
        if dibujar and (cant_atrib == 2):        
            adalinePlot(P, T2, W, b, 'Iteración: ' + str(ite) + ' - Error promedio: ' + str(error_prom))
        
    return (W, b, ite, error_prom)


def adalinePredict(X, W, b, funcion, umbral):
    """
    Predice la clase de las muestras (puntos) ingresados a partir del array
    de pesos del perceptrón entrenado.
    Por ser un modelo de una sola neurona puede predecir solo entre 
    dos clases 0 o 1
    Entradas 
       X: Lista con las muestras (puntos) a predecir
       W: Array con los pesos del perceptrón entrenadas previamente y que se 
          usarán para hacer la predición de la clase de cada muestra (punto) de X
       b: Valor del bias (W0) del perceptrón entrenado
       Funcion: La función de activación (la misma usada en el entrenamiento)
       Umbral: Define el umbral apartir del cual (>) se considera que la clase
               es un uno
    Salida
       salidaNorm: Lista con la clase predicha de cada uno de las muestras (puntos)
                   recibidos
    """
    salidaNorm = []
    if X.shape[1] != W.shape[1]:
        print("No se puede predecir, pues la cantidad de variables de entrada \
               (columnas de X) debe ser igual a la cantidad de pesos del peceptrón \
               (columnas de W")
    else:
        salida = []
        for obj in X:
            sumatoria = 0
            i = 0
            for itm in W:
                j = 0
                for ind in itm:
                    sumatoria = sumatoria + (ind * obj[[j][i]])
                    j = j + 1
                i = i + 1
            sumatoria = sumatoria + b[0]
            
            #funcion = RedesNeuronales.logsig(sumatoria)
            funcionResult = eval(funcion + '(sumatoria)')
            
            salida.append(funcionResult)
        
        # Aplica umbral para determinar la clase
        salidaNorm = []
        for elmt in salida:
            if elmt > umbral:
                salidaNorm.append(1)
            else:
                salidaNorm.append(0)
        
    return salidaNorm
    
        

#------------------------------------------------------------------------------
# BPN - Back Propagation Neuronal Network
#------------------------------------------------------------------------------
def bpnTrain(P, T, T2, ocultas, alfa, momento, fun_oculta, fun_salida, max_itera, cota_error, dibujar):
    """
    Parámetros:
           P: es una matriz con los datos de los patrones con los cuales
               entrenar la red neuronal. Los ejemplos deben estar en columnas.    @kSelta creo que se refiere a que deben estar en FILAS
           T: es una matriz con la salida esperada para cada ejemplo. Esta matriz 
               debe tener tantas filas como neuronas de salida tenga la red       @kSelta creo que se refiere a COLUMNAS
           T2: clases con su valor original (0 .. n-1) (Solo es utilizado para graficar)
           ocultas: la cantidad de neuronas ocultas que tendrá la red    
           alfa: velocidad de aprendizaje
           momento: término de momento
           fun_oculta: función de activación en las neuronas de la capa oculta
           fun_salida: función de activación en las neuronas de la capa de salida
           MAX_ITERA: la cantidad máxima de iteraciones en las cuales se va a
               ejecutar el algoritmo
           cota_error: error mínimo aceptado para finalizar con el algoritmo
           dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
               ejemplos y las rectas discriminantes.
    
    Devuelve:
           w_O: la matriz de pesos de las neuronas de la capa oculta
           b_O: vector de bias de las neuronas de la capa oculta
           w_S: la matriz de pesos de las neuronas de la capa de salida
           b_S: vector de bias de las neuronas de la capa de salida
           ite: número de iteraciones ejecutadas durante el algoritmo
           error_prom: errorPromedio finalizado el algoritmo
    
    Ejemplo de uso:
           (w_O, b_O, w_S, b_S, ite, error_prom) = bpnTrain(P, T, T2, 10, 0.25, 1.2, 'logsig', 'tansig', 25000, 0.001, True);
           
    """
    
    (cant_patrones, cant_atrib) = P.shape
    (cant_patrones, cant_salidas) = T.shape
    
    w_O = np.random.rand(ocultas, cant_atrib) - 0.5
    b_O = np.random.rand(ocultas,1) - 0.5
    w_S = np.random.rand(cant_salidas, ocultas) - 0.5
    b_S = np.random.rand(cant_salidas,1) - 0.5
        
    return bpnTrain_con_pesos(P, T, T2, ocultas, alfa, momento, fun_oculta, fun_salida, max_itera, cota_error, dibujar, w_O, b_O, w_S, b_S)
   

def bpnTrain_con_pesos(P, T, T2, ocultas, alfa, momento, fun_oculta, fun_salida, max_itera, cota_error, dibujar, w_O, b_O, w_S, b_S):
    (cant_patrones, cant_atrib) = P.shape
    
    momento_w_S = np.zeros(w_S.shape)
    momento_b_S = np.zeros(b_S.shape)
    momento_w_O = np.zeros(w_O.shape)
    momento_b_O = np.zeros(b_O.shape)

    ite = 0;
    error_prom = cota_error + 1
    ultimoError = cota_error +1
    anteultimoError = cota_error +2
    
    print('iteracion -- ', 'error_prom -- ', 'diferencia de error ultimas 2 ejecuciones')
    
    #while (ite < max_itera) and (abs(ultimoError - anteultimoError) > cota_error): # TODO: Preguntar por qué la condición
    #    suma_error = 0                                                        # de salida es esta resta y no el error promedio
    #    for p in range(cant_patrones):                                        # o la suma del error cuadrático   
    while (ite < max_itera) and (error_prom > cota_error):    
        suma_error = 0
        for p in range(cant_patrones):
            neta_oculta = w_O.dot(P[p,:][np.newaxis].T) + b_O
            salida_oculta = eval(fun_oculta + '(neta_oculta)')
            neta_salida = w_S.dot(salida_oculta) + b_S
            salida_salida = eval(fun_salida + '(neta_salida)')
           
            error_ejemplo = T[p,:] - salida_salida.T[0]
            suma_error = suma_error + np.sum(error_ejemplo**2)

            delta_salida = error_ejemplo[np.newaxis].T * eval('d' + fun_salida + '(neta_salida)')
            delta_oculta = eval('d' + fun_oculta + '(neta_oculta)') * w_S.T.dot(delta_salida)
            
            w_S = w_S + alfa * delta_salida * salida_oculta.T + momento * momento_w_S
            b_S = b_S + alfa * delta_salida + momento * momento_b_S
             
            w_O = w_O + alfa * delta_oculta * P[p,:] + momento * momento_w_O
            b_O = b_O + alfa * delta_oculta + momento * momento_b_O
           
            momento_w_S = alfa * delta_salida * salida_oculta.T + momento * momento_w_S
            momento_b_S = alfa * delta_salida + momento * momento_b_S            
            
            momento_w_O = alfa * delta_oculta * P[p,:].T + momento * momento_w_O
            momento_b_O = alfa * delta_oculta + momento * momento_b_O
            
        error_prom = suma_error / cant_patrones
        
        anteultimoError = ultimoError
        ultimoError = error_prom
        
        ite = ite + 1
        print(ite, error_prom, abs(ultimoError - anteultimoError))   
        
        if dibujar and (cant_atrib == 2):        
            bpnPlot(P, T2, w_O, b_O, 'Iteración: ' + str(ite) + ' - Error promedio: ' + str(error_prom))
        
    return (w_O, b_O, w_S, b_S, ite, error_prom)


def bpnPlot(P, T, W, b, title = ''):
    marcadores = {0:('+','b'), 1:('o','g'), 2:('x', 'y'), 3:('*', 'm'), 4:('.', 'r'), 5:('+', 'k')}
    plt.clf()
    
    #Ejemplos
    for class_value in np.unique(T):
        x = []
        y = []
        for i in range(len(T)):
            if T[i] == class_value:
                x.append(P[i, 0])
                y.append(P[i, 1])
        plt.scatter(x, y, marker=marcadores[class_value][0], color=marcadores[class_value][1])
    
    #ejes
    minimos = np.min(P, axis=0)
    maximos = np.max(P, axis=0)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    plt.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    
    #rectas discriminantes
    x1 = minimos[0]
    x2 = maximos[0]
    (neuronas, patr) = W.shape
    for neu in range(neuronas):
        m = W[neu,0] / W[neu,1] * -1
        n = b[neu] / W[neu,1] * -1
        y1 = x1 * m + n
        y2 = x2 * m + n
        plt.plot([x1, x2],[y1, y2], color='r')
    
    plt.title(title)
    
    plt.draw()
    plt.pause(0.00001) 



#------------------------------------------------------------------------------
# SOM - Self Organizing Map
#------------------------------------------------------------------------------ 
def somTrain(P, filas, columnas, alfa_inicial, vecindad, fun_vecindad, sigma, ite_reduce, dibujar):
    """
    La forma inicial de la red es rectangular de tamaño filas x columnas

    Parámetros:
           P: es una matriz con los datos de los patrones con los cuales
               entrenar la red neuronal. 
           filas: la cantidad de filas del mapa SOM
           columnas: la cantidad de columnas del mapa SOM
           alfa_inicial: velocidad de aprendizaje inicial
           vecindad: vecindad inicial
           fun_vecindad: función para determinar la vecindad (1: lineal, 2: sigmoide)
           sigma: ancho de la campana (solo para vecindad sigmoide)
           ite_reduce: la cantidad de iteraciones por cada tamaño de vecindad
               (la cantidad de iteraciones total sera: total = ite_reduce * (vecindad+1))
           dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
               ejemplos y el mapa SOM.
    
    Devuelve:
           w_O: la matriz de pesos de las neuronas competitivas
    
    Ejemplo de uso:
           (w_O) = somTrain(P, filas, columnas, alfa, vecindad, fun_vecindad, sigma, ite_reduce, True);
    """
    (cant_patrones, cant_atrib) = P.shape

    ocultas = filas * columnas    
    w_O = np.random.rand(ocultas, cant_atrib) - 0.5
    
    w_O = np.ones((ocultas, cant_atrib)) * 0
    
    pasos = somLinkdist(filas, columnas)
    
    max_ite = ite_reduce * (vecindad + 1)
    ite = 0;
    
    while (ite < max_ite):
        alfa = alfa_inicial * (1 - ite / max_ite)
        for p in range(cant_patrones): 
            distancias = -np.sqrt(np.sum((w_O-(P[p,:])*np.ones((ocultas,1)))**2,1))
            ganadora = np.argmax(distancias)
            fila_g = int(np.floor(ganadora / columnas))
            columna_g = int(ganadora % columnas)

            for f in range(filas):
               for c in range(columnas):
                   if(pasos[fila_g, columna_g, f, c] <= vecindad):
                       if fun_vecindad == 1:
                           gamma = 1
                       else:
                           gamma = np.exp(- pasos[fila_g, columna_g, f, c] / (2*sigma))
              
                       n = f * columnas + c
                       w_O[n,:] = w_O[n,:] + alfa * (P[p,:] - w_O[n,:]) * gamma
            
        ite = ite + 1
        
        if (vecindad >= 1) and ((ite % ite_reduce)==0):
            vecindad = vecindad - 1;
        
        if dibujar and (cant_atrib == 2):        
            somPlot(P, None, w_O, filas, columnas, pasos, 'Iteración: ' + str(ite))    
    return (w_O)


def somTrainOL(P, T, T_O, w, filas, columnas, alfa, max_ite, dibujar):
    """
    Entrenamiento de la capa de salida (OL) que se aplica sobre la capa 
    competitiva entrenada previamente con la función somTrain
    Parámetros:
           P: es una matriz con los datos de los patrones con los cuales
              entrenar la red neuronal. 
           T: es una matriz con la salida esperada para cada ejemplo. Esta matriz 
              debe tener tantas filas como neuronas de salida tenga la red
           T_O: clases con su valor original (0 .. n-1) (Solo es utilizado para graficar)
           w: es la matriz de pesos devuelta por la función trainSOM
           filas: la cantidad de filas del mapa SOM
           columnas: la cantidad de columnas del mapa SOM
           alfa: velocidad de aprendizaje 
           max_ite: la cantidad de iteraciones del entrenamiento
           dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
                    ejemplos y el mapa SOM.
    Devuelve:
           w_S: la matriz de pesos de las neuronas de la capa de salida
    
    Ejemplo de uso:
           (w_S) = somTrainOL(P, T_matriz.T, T, w, filas, columnas, alfa, 100, True);
    """
    
    (cant_patrones, cant_atrib) = P.shape
    (cant_patrones, salidas) = T.shape   
    ocultas = filas * columnas
    
    pasos = somLinkdist(filas, columnas)
    w_S = np.random.rand(salidas, ocultas) - 0.5
    
    ite = 0;
    while ( ite <= max_ite ):
        for p in range(cant_patrones): 
            distancias = -np.sqrt(np.sum((w-(P[p,:])*np.ones((ocultas,1)))**2,1))
            ganadora = np.argmax(distancias)
       
            w_S[:, ganadora] = w_S[:, ganadora] + alfa * (T[p, :] - w_S[:, ganadora])
    
        ite = ite + 1
        
    if dibujar and (cant_atrib == 2):
        somPlot(P, T_O, w, filas, columnas, pasos, 'Fin')
    return (w_S)


def somUmatrix(w, filas, columnas):
    """
    Cálcula la matriz de distancia entre las neuronas del SOM y gráfica el mapa
    de calor.
    Parámetros:
           w: es la matriz de pesos devuelta por la función trainSOM
           filas: la cantidad de filas del mapa SOM
           columnas: la cantidad de columnas del mapa SOM
    Devuelve:
           umat: la matriz de distancias del SOM
    Ejemplo de uso:
           umat = somUmatrix(w, filas, columnas)
    """
    (ncen, atributos) = w.shape 
    umat = np.zeros((filas*2-1, columnas*2-1))
    
    for f in range(filas):
        for c in range(columnas):
            ff = f*2
            cc= c*2
            n1 = f * columnas + c 
            suma = 0
            n=0
            
            n2 = f * columnas + (c+1)
            if(cc < (columnas*2-2)):
                umat[ff, cc+1] = np.sqrt(np.sum((w[n1,:]-w[n2,:])**2))                
                suma = suma + umat[ff, cc+1]
                n=n+1
            n2 = (f+1) * columnas + c
            if(ff < (filas*2-2)):
                umat[ff+1, cc] = np.sqrt(np.sum((w[n1,:]-w[n2,:])**2))                
                suma = suma + umat[ff+1, cc]
                n=n+1            
            if(n==2):
                umat[ff+1, cc+1] = suma / 2
                suma = suma + umat[ff+1, cc+1]
                n=n+1
            if(n>0):
                umat[ff, cc] = suma / n
    umat[filas*2-2, columnas*2-2] = (umat[filas*2-3, columnas*2-2] + umat[filas*2-2, columnas*2-3]) / 2
            
    plt.figure(1)
    plt.imshow(umat, cmap='hot')
    plt.show()
    return umat


def somDendrograma(matriz, T):
    (filas, columnas) = matriz.shape
    labels=[]
    for f in range(filas):
        labels.append(str(int(T[f])))
        for c in range(columnas):
            if(f==c):
                matriz[f,c] = 0
            else:
                if(matriz[f,c] == 0):
                    matriz[f,c] = 2
                else:
                    matriz[f,c] = 1 / matriz[f,c]
                    
    
    
    dists = squareform(matriz)
    linkage_matrix = linkage(dists, "single")
    dendrogram(linkage_matrix, labels=labels)
    plt.title("Dendrograma")
    plt.show()
    

def somPlot(P, T, W, filas, columnas, pasos, title):
    marcadores = {0:('+','b'), 1:('o','g'), 2:('x', 'y'), 3:('*', 'm'), 4:('.', 'r'), 5:('+', 'k')}
    plt.figure(0)
    plt.clf()
    
    #Ejemplos
    x = []
    y = []
    if(T is None):
        for i in range(P.shape[0]):
            x.append(P[i, 0])
            y.append(P[i, 1])
        plt.scatter(x, y, marker='+', color='b')
    else:
        colores = len(marcadores)
        for class_value in np.unique(T):
            x = []
            y = []
            for i in range(len(T)):
                if T[i] == class_value:
                    x.append(P[i, 0])
                    y.append(P[i, 1])
            plt.scatter(x, y, marker=marcadores[class_value % colores][0], color=marcadores[class_value % colores][1])
    
    #ejes
    minimos = np.min(P, axis=0)
    maximos = np.max(P, axis=0)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    plt.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    
    #centroides
    (neuronas, patr) = W.shape
    for neu in range(neuronas):
        plt.scatter(W[neu,0], W[neu,1], marker='o', color='r')
        
    #conexiones
    if(pasos is not None):
        for f in range(filas):
            for c in range(columnas):
                n1 = f*columnas + c
                for ff in range(filas):
                    for cc in range(columnas):
                        if(pasos[f, c, ff, cc] == 1):
                            n2 = ff*columnas + cc
                            plt.plot([W[n1, 0], W[n2, 0]], [W[n1, 1], W[n2, 1]], color='r')                   
    
    plt.title(title)
    plt.draw()
    plt.pause(0.00001)
    
    
def somLinkdist(filas, columnas):
    pasos = np.zeros((filas, columnas, filas, columnas))
    for f in range(filas):
        for c in range(columnas):
            for ff in range(filas):
                for cc in range(columnas):
                    pasos[f, c, ff, cc] = abs(f-ff) + abs(c-cc)
    return pasos



    
