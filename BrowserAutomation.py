# -*- coding: utf-8 -*-
"""
    BrowserAutomation.py
    Created on Feb 9 2020

    Contiene las funciones para realizar automatización de tareas en web 
    browsers de una manera más limpia.

    @author: @jaimeHMol
"""

# IMPORTS GLOBALES
# ----------------
#import time
#from datetime import datetime, timedelta
#import json
#import requests

# Importa la libreria selenium que debió ser instalada previamente
# usando pip en el comand prompt
from selenium import webdriver


# CONSTANTES GLOBALES
# -------------------



# VARIABLES GLOBALES
# -------------------



# MAIN
# -----
# Funciones para hacer más prolija la navegación controlada desde Python
def initialize(driverPath, url):
    """ Inicializar el driver a utilizar en la automatización a realizar
        If driverPath = "" will try to obtain the driver from the PATH
    """
    if driverPath:
        chm_driver = webdriver.Chrome(driverPath)
    else:
        chm_driver = webdriver.Chrome() # HINT: Could not work if the selenium chrome driver is not in the PATH
    
    chm_driver.get(url)
    return chm_driver


def navigate(chm_driver, url):
    """ Acceder a la url ingresada
    """
    chm_driver.get(url)


def scripting(chm_driver, script):
    """ Ejecuta el script ingresado. 
        
        Usualmente utilizado para ingresar valores en los formularios, campos, etc.
        Ejemplo: scripting(chm_driver, "document.getElementById('txtUsuario').value = '" + usuario + "'")
        scripting(chm_driver, "document.getElementsByName('loginUsername')[0].value = '" + USER + "'")
        
        También es usado para ejecutar funcionalidades implementadas en javascript
        Ejemplo: scripting("__doPostBack('ctl00$ContentPlaceHolder1$txtHoras')")        
    """
    chm_driver.execute_script(script)


def scripting_return(chm_driver, script):
    """ Ejecuta el script ingresado y devuelve el elemento resultado
    """
    element = chm_driver.execute_script('return ' + str(script))
    return element


def selecting_by_id(chm_driver, selector):
    """ Selecciona un elemento buscandolo por id
    """
    element = chm_driver.find_elements_by_id(selector)
    return element


def selecting_by_name(chm_driver, selector):
    """ Selecciona un elemento buscandolo por name
    """
    element = chm_driver.find_elements_by_name(selector)
    return element


def selecting_by_classname(chm_driver, selector):
    """ Selecciona un elemento buscandolo por classname
    """
    element = chm_driver.find_element_by_class_name(selector)
    return element


def selecting_by_tag(chm_driver, selector):
    """ Selecciona un elemento buscandolo por el tag ingresado
    """
    element = chm_driver.find_elements_by_tag_name(selector)
    return element


def entering_by_id(chm_driver, selector, value):
    """ Ingresa el valor a un elemento buscandolo por id
    """
    chm_driver.execute_script("document.getElementById('" + selector + 
                                           "').value = '" + value + "'")


def entering_by_name(chm_driver, selector, value):
    """ Ingresa el valor a un elemento buscandolo por name
        HINT: Ingresará el valor deseado (value) SOLO en el primer elemento 
              que tenga la nombre ingresado (selector)
    """
    chm_driver.execute_script("document.getElementsByName('" + selector +
                                           "')[0].value = '" + value + "'")
    

def entering_by_classname(chm_driver, selector, value):
    """ Ingresa el valor a un elemento buscandolo por classname
        HINT: Ingresará el valor deseado (value) SOLO en el primer elemento 
              que tenga la clase ingresada (selector)
    """
    chm_driver.execute_script("document.getElementsByClassName('" + selector +
                                                "')[0].value = '" + value + "'")


def entering_by_tag(chm_driver, selector, value):
    """ Ingresa el valor a un elemento buscandolo por el tag ingresado
        HINT: Ingresará el valor deseado (value) SOLO en el primer elemento 
              que tenga el tag ingresado (selector)
    """
    chm_driver.execute_script("document.getElementsByTagName('" + selector +
                                              "')[0].value = '" + value + "'")


def clicking_by_id(chm_driver, selector):
    """ Hace click en un elemento buscandolo por id
    """
    element = chm_driver.find_elements_by_id(selector)
    element[0].click()


def clicking_by_name(chm_driver, selector):
    """ Hace click en un elemento buscandolo por name
    """
    element = chm_driver.find_elements_by_name(selector)
    element[0].click()
        
    
def clicking_by_classname(chm_driver, selector):
    """ Hace click en un elemento buscandolo por classname
    """
    element = chm_driver.find_elements_by_class_name(selector)
    element[0].click()


def clicking_by_other_tag(chm_driver, tagName, selector):
    """ Hace click en un elemento buscandolo por el tag ingresado
    """
    element = chm_driver.find_elements_by_xpath('//*[@'+ tagName
                                                + '="' + selector + '"]')
    element[0].click()
    

def page_source(chm_driver):
    """ Recupera el código fuente de la página web
    """
    elem = chm_driver.page_source
    return elem


def close(chm_driver):
    """ Cierra el navegador
    """
    chm_driver.close()
    chm_driver.quit()


def wait(chm_driver, wtime):
    """ Pausa el navegador, wtime en segundos
    """
    chm_driver.implicitly_wait(wtime)

