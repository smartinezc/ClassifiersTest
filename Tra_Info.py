#!/usr/bin/env python
# coding: utf-8

# In[205]:
from io import open
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from imblearn.under_sampling import RepeatedEditedNearestNeighbours 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('sintetico.txt', sep=",", header=None,skiprows=3) 

print("Numero de vecinos cercanos a eliminar")
k = int(input())

valores = data.values  # Arreglo de Valores

clase = valores[:, -1] # Seleccionamos las clases
valores = valores[:, :-1] # Seleccionamos los atributos

valores.astype(int)
clase.astype(float)
X = valores[:, :2]  # Seleccionamos los dos primeros atributos
y = clase
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(1, figsize=(8, 6))
plt.clf()

# Graficando puntos a eliminar
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.title("Dataset con Selección de Datos a Eliminar")
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
renn = RepeatedEditedNearestNeighbours(n_neighbors=k)
X_res, y_res = renn.fit_resample(X, y)
plt.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=y_res,
           s=25, edgecolor='k', cmap=plt.cm.coolwarm)

print('Dataset Original %s' % Counter(y))
print('Dataset Suavizado %s' % Counter(y_res))

plt.figure(2, figsize=(8, 6))
plt.clf()
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=y_res,
           s=25, edgecolor='k', cmap=plt.cm.coolwarm)
plt.title("Suavizado de Frontera con K Vecinos Cercanos(k = %i)" % (k))
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=101)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
y_pred = knn.predict(X)
print("Puntaje de presición Clasificador KNN con fronteras suavizadas:",metrics.accuracy_score(y, y_pred)*100)

X_trains, X_tests, y_trains, y_tests = train_test_split(X, y, test_size=0.3, random_state=101)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_trains, y_trains)
preds = knn.predict(X_tests)
y_preds = knn.predict(X)
print("Puntaje de presición Clasificador KNN sin suavizado:",metrics.accuracy_score(y, y_preds)*100)

bayes = GaussianNB()
bayes.fit(X_test, y_test)
y_pred = bayes.predict(X_test)
print("Puntaje de presición Gaussian Naive Bayes con fronteras suavizadas:",metrics.accuracy_score(y_test, y_pred)*100)

bayes2 = GaussianNB()
bayes2.fit(X, y)
y_pred2 = bayes2.predict(X)
print("Puntaje de presición Gaussian Naive Bayes sin suavizado:",metrics.accuracy_score(y, y_pred2)*100)

# %%
