# coding: utf-8

import SuavizarFronteraKNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import metrics

#Leer los datos del archivo
datOriginales = pd.read_csv('sintetico.txt', sep=',', header=None, skiprows=3)
atributosOr = datOriginales.values[:, :-1][:, 0:2] #Almacena los atributos de cada dato del set original
clasesOr = datOriginales.values[:, -1]     #Almacena la clase de cada dato del set original

#Crea un objeto de la clase encargada de suavizar los datos con los datos del archivo de texto especificado
suaviza = SuavizarFronteraKNN.Suavizado("sintetico.txt")

#Suaviza la frontera de los datos para los atributos [atr1, atr2] y el K especificados
datSuavizados = pd.DataFrame(suaviza.suavizarDatosKNN([0, 1], 10))
atributosSu = datSuavizados.values[:, :-1] #Almacena los atributos de cada dato del set suavizado
clasesSu = datSuavizados.values[:, -1]     #Almacena la clase de cada dato del set suavizado

#Separar datos para entrenamiento y validación para el set original
#(Se usa random_state = 4444 para que el set resultante sea constante)
atrEntrenoOr, atrValidaOr, claEntrenoOr, claValidaOr = train_test_split(atributosOr, clasesOr,
                                                                        test_size = 0.2, random_state = 4444)

#Separar datos para entrenamiento y validación para el set suavizado
#(Se usa random_state = 4444 para que el set resultante sea constante)
atrEntrenoSu, atrValidaSU, claEntrenoSu, claValidaSu = train_test_split(atributosSu, clasesSu,
                                                                        test_size = 0.2, random_state = 4444)

#Crear una instancia del clasificador KNN para los 5 vecinos más cercanos
knnClf = neighbors.KNeighborsClassifier(n_neighbors=5)

#Clasificación de los datos originales
knnClf.fit(atrEntrenoOr, claEntrenoOr)            #Entrena el modelo KNN con los datos de entrenamiento originales
claPredecidaOrKNN = knnClf.predict(atrValidaOr)   #Clasifica los datos introducidos y retorna la clase de los datos
print("---- Clasificados con KNN - Set Original ----------------------------------")
print("Precisión: {:0.3f}%".format(metrics.accuracy_score(claValidaOr, claPredecidaOrKNN)*100))

#Clasificación de los datos suavizados
knnClf.fit(atrEntrenoSu, claEntrenoSu)            #Entrena el modelo KNN con los datos de entrenamiento originales
claPredecidaSuKNN = knnClf.predict(atrValidaSU)   #Clasifica los datos introducidos y retorna la clase de los datos
print("---- Clasificados con KNN - Set Suavizado ----------------------------------")
print("Precisión: {:0.3f}%".format(metrics.accuracy_score(claValidaSu, claPredecidaSuKNN)*100))



#Crear una instancia del clasificador Naives-Bayes
nBClf = naive_bayes.GaussianNB()

#Clasificación de los datos originales
nBClf.fit(atrEntrenoOr, claEntrenoOr)          #Entrena el modelo Naive-Bayes con los datos de entrenamiento originales
claPredecidaOrNB = nBClf.predict(atrValidaOr)  #Clasifica los datos introducidos y retorna la clase de los datos
print("---- Clasificados con Naive-Bayes - Set Original ---------------------------")
print("Precisión: {:0.3f}%".format(metrics.accuracy_score(claValidaOr, claPredecidaOrNB)*100))

#Clasificación de los datos suavizados
nBClf.fit(atrEntrenoSu, claEntrenoSu)          #Entrena el modelo Naive-Bayes con los datos de entrenamiento originales
claPredecidaOrNB = nBClf.predict(atrValidaSU)  #Clasifica los datos introducidos y retorna la clase de los datos
print("---- Clasificados con Naive-Bayes - Set Suavizado --------------------------")
print("Precisión: {:0.3f}%".format(metrics.accuracy_score(claValidaSu, claPredecidaOrNB)*100))
