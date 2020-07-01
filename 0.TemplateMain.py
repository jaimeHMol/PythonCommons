# -*- coding: utf-8 -*-
"""
    <NombreProgramaPython>.py
    Created on <Fecha>

    Plantilla con la estructura básica ordenada de un programa tipo Main de Python.

    @author: @jaimeHMol
"""

# IMPORTS GLOBALES
# ----------------
import os
import sys
from pathlib import Path # Para almacenar rutas de la misma forma sin importar el sistema operativo
import logging as log


# CONSTANTES GLOBALES
# -------------------
# *** Para construir una aplicación: ***
#ROOTAPPPATH      = Path(os.getcwd()) # Si el script es invocado desde línea de comandos (windows)
#RESOURCESAPPPATH = ROOTAPPPATH / "Resources"
#LIBAPPPATH       = ROOTAPPPATH / "Lib"
#DATAPATH         = Path("") # <-- LLENAR AQUI
#APPNAME          = "AppName"
#ICONNAME         = APPNAME + ".ico"
#FRAMETITLE       = APPNAME + "FrameTittle"
#LOGFILENAME      = ROOTAPPPATH / (APPNAME + ".log")
# ***************************************

# *** Para construir un script: ***
#HOMEPATH   = Path(os.getcwd()) # Si el script es invocado desde línea de comandos

# Windows: (Unidad P:/ usando bat para mapear unidades virtuales)
HOMEPATH    = Path("P:/") # <-- LLENAR AQUI

# MACOS: (Configurar alias permanente a esta ruta)
#HOMEPATH   = Path("/home/user/Projects/Python/") # <-- LLENAR AQUI

# Linux: (Configurar alias permanente a esta ruta)
#HOMEPATH   = Path("/home/pi/Projects/Python/") # <-- LLENAR AQUI

COMMONPATH    = HOMEPATH / "Common"
sys.path.insert(0, str(COMMONPATH))

CONFIGPATH     = HOMEPATH / "0.Config"
DATAPATH       = Path("") # <-- LLENAR AQUI
OUTPUTPATH     = HOMEPATH / "Work"
SCRIPTNAME     = "ScriptName" # <-- LLENAR AQUI
OUTPUTFILENAME =  OUTPUTPATH / (SCRIPTNAME + "_out.txt")

LOGLEVEL    = 20 # 10->DEBUG, 20->INFO, 30->WARNING, 40->ERROR, 50->CRITICAL
LOGTERMINAL = True
LOGFILE     = False
LOGFILENAME = HOMEPATH / (SCRIPTNAME + ".log")
LOGFILEMODE = 'w' # a: append, w: overwrite
# ************************************

DATEFORMAT       = "%Y%m%d"
TIMESTAMPFORMAT  = "%Y-%m-%d %H:%M:%S"
NAMEPARAMINIFILE = "paramIni.xml"
NAMECONFIGFILE   = "config.xml"


# VARIABLES GLOBALES
# ------------------



# Imports propios
# ---------------
# Here you can import modules from the XXXLIBPATH constant defined before

# *** Install a module with PIP from a Python script
#import subprocess
#subprocess.check_call([sys.executable, "-m", "pip", "install", "httpimport"])

# *** Importar módulo desde un repositorio de GitHub (puede extenderse a cualquier endpoint HTTP) ***
#import httpimport
#from httpimport import github_repo
#
## My own Python Utils library
#with httpimport.github_repo("jaimehmol", "PythonCommons", module='Utils', branch = 'master'):
#    import Utils
    
    
# FUNCIONES
# ---------                
def genRandomData(numVariables, numMuestras):
    """ Function documentation
    """    
    from sklearn.datasets import samples_generator
    X, y = samples_generator.make_classification(n_samples = numMuestras, 
                                                 n_features = numVariables,
                                                 n_informative=2, 
                                                 n_redundant=0, 
                                                 random_state=77)
    return X, y


def loggingInitialization(scriptName, timeStampFormat,
                          logLevel, 
                          terminal, 
                          file, fileName, fileMode):
    """ Set the initial configuration for the logging "strategy" to be used on the script
        requires import logging as log
        
        scriptName: Name of the script that will log
        timeStampFormat: Date mask in which you want to see the logs
        logLevel: Log level to be send to the handlers (10->DEBUG, 20->INFO, 30->WARNING, 40->ERROR, 50->CRITICAL)
    
        terminal: Is a boolean to indicate if you want to log trhough the terminal
        file: Is a boolean to indicate if yoy want to log trhough a file
        You can request just one or both, but must select at least one 
        
        fileName: If you select file, it is the output log file path and name
        fileMode: If you select file, mode to write the log file. a-> append. w-> overwrite 
    """
    
    buildFormat = "%(asctime)s - " + scriptName + " [%(levelname)s]: %(message)s. (%(name)s.%(lineno)d)"
    
    if not terminal and not file:
        raise ValueError("You must define at least one way (handler) to place the logs")
        
    elif terminal and file :
        log.basicConfig(format   = buildFormat,
                        datefmt  = timeStampFormat,
                        level    = logLevel,
                        handlers = [log.StreamHandler(), # Terminal (stderr)
                                    log.FileHandler(fileName, mode = fileMode)]) 
        
    elif terminal:
        log.basicConfig(format   = buildFormat,
                        datefmt  = timeStampFormat,
                        level    = logLevel,
                        handlers = [log.StreamHandler()]) 

    
    elif file:
        log.basicConfig(format   = buildFormat,
                        datefmt  = timeStampFormat,
                        level    = logLevel,
                        handlers = [log.FileHandler(fileName, mode = fileMode)]) 

    # To remove a handler located on index=0
    #log.getLogger('').handlers.pop(0)
         
    
def loggingChangeLogLevel(index, logLevel):
    """ Changing the log level of an existing handler configured using the 
        loggingInitialization function.
        
        index: represent the position in which the handler was created, usual case:
            0 -> Terminal handler
            1 -> File handler (If it was defined)
        logLevel: Log level to be send to the handlers (10->DEBUG, 20->INFO, 30->WARNING, 40->ERROR, 50->CRITICAL)            
    """
    log.getLogger('').handlers[index].setLevel(logLevel)


# CLASES
# ------
# Clase de ejemplo:
class nodo:
    """ Define la características (variables) y acciones (métodos) de un nodo
        de un arbol binario
    """
    ident = 0
    valor = " "
    hijoIzq = 0
    hijoDer = 0
    padre = 0

    def __init__(self, ident, valor, padre, hijoIzq, hijoDer):
        self.ident = ident
        self.valor = valor
        self.hijoIzq = hijoIzq
        self.hijoDer = hijoDer
        self.padre = padre


    def actualizaNodo(self, valor, padre, hijoIzq, hijoDer):
        self.valor = valor
        self.hijoIzq = hijoIzq
        self.hijoDer = hijoDer
        self.padre = padre



# MAIN
# ----
if __name__ == "__main__":

    loggingInitialization(scriptName = SCRIPTNAME, timeStampFormat = TIMESTAMPFORMAT,
                          logLevel = LOGLEVEL, 
                          terminal = LOGTERMINAL, 
                          file = LOGFILE, fileName = str(LOGFILENAME), fileMode = LOGFILEMODE)
    log.info("Prints para depurar")
    
    # Generar datos aleatorios para probar cosas
    # X, y = genRandomData(numVariables, numMuestras)
