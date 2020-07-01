# -*- coding: utf-8 -*-
"""
    Utils.py
    Created on Feb 9 2020

    Contains my general and reusable util functions.

    @author: @jaimeHMol
"""

# IMPORTS GLOBALES
# ----------------
import sys
import os
import subprocess


# CONSTANTES GLOBALES
# -------------------



# VARIABLES GLOBALES
# ----------------



# MAIN
# -----
def cantDiasAnho(year):
    """ Detecta si es año bisiesto y devuelve la cantidad de dìas del año
    """
    if year % 400 == 0:
        return 366
    if year % 100 == 0:
        return 365
    if year % 4 == 0:
        return 366
    else:
        return 365
    

def install(package):
    """ To install pip modules right from Python script.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    

def makeDir(path):
    """ Crea el directorio de un archivo que se va a crear posteriormente
    """
    path = os.path.dirname(path)
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
   
     
def genRandomData(numVariables, numMuestras):
    """ Genera datos al azar para las cantidad de variables definidas con el 
        parametro numVariables (columnas) que debe ser mayor a 1, con la 
        cantidad de muestras definidas con el parametro numMuestras (filas), 
        que producen una variable de salida con la clase binaria (1 o 0) a la 
        que pertenece cada muestra.
        Por lo tanto las salidas son: la matriz X de todas las muestras con sus
        respectivos valores para cada variable, y la salida y con la clase
        de cada muestra.
    """    
    from sklearn.datasets import samples_generator
    X, y = samples_generator.make_classification(n_samples = numMuestras, 
                                                 n_features = numVariables,
                                                 n_informative=2, 
                                                 n_redundant=0, 
                                                 random_state=77)
    return X, y