
import numpy as np
import matplotlib.pyplot as plt
import io
from collections import Counter


class Suavizado():
    """Clase que recibe una base de datos y suaviza la frontera de los datos para un par de atributos especificado"""

    def __init__(self, rutaDatos):
        """Crea una lista donde cada elemento es una matriz de número de atributos x número de datos por clase"""

        #Abrir archivo de texto para lectura
        archivoTexto = io.open(rutaDatos, 'r')
        #Guarda en una lista las líneas del documento
        lineas = archivoTexto.readlines()

        #Guarda el número de datos, el número de atributos y el número de clases
        self.numLineas = int(lineas[0])
        self.numAtributos = int(lineas[1])
        self.numClases = int(lineas[2])

        #Crea una lista donde cada elemento es una matriz de número de atributos x número de datos por clase
        self.data = [np.zeros(self.numAtributos) for x in range(self.numClases)]
        for l in range(3, self.numLineas+3):
            clase = int(lineas[l].split(',')[self.numAtributos])
            datLinea = lineas[l].split(',')[0:self.numAtributos]
            datLinea = [float(i) for i in datLinea]

            #Si es el primer dato de la clase borra la matriz de ceros, si no lo agrega a la matriz
            if not self.data[clase].any():
                self.data[clase] = np.array(datLinea)
            else:
                self.data[clase] = np.vstack([self.data[clase], datLinea])

        #Cierra el archivo de texto
        archivoTexto.close()

    def darDatosEntrada(self, clase=-1):
        """Retorna los datos de entrada de la clase especificada por parámetro. Si el parámetro 'clase' no
         se especifica, se retorna los datos de todas las clases"""
        if clase == -1:
            return self.data
        else:
            return self.data[clase]

    def suavizarDatosKNN(self, atributos, k=-1):
        """Retorna una lista donde cada elemento contiene una lista con los valores de los atributos especificados
        por parámetro y la clase de ese dato después de suavizar la frontera"""

        #Si no se especifica un k, se emplea el número de clases más uno, así siempre habrá una clase con mayoría
        if k == -1:
            k = self.numClases+1

        #Itera sobre las clases de los datos y almacena en d cada punto
        d = []
        for cl,dat in enumerate(self.data):
            numF = dat.shape[0]

            #Itera sobre los datos de la clase actual según los atributos especificados
            for f in range(numF):
                punto = dat[f, atributos]
                #d es una lista de 3 columnas: las coordenadas de los atributos y la clase
                d.append([punto[0], punto[1], cl])

        #Itera sobre los puntos
        for inD,pD in enumerate(d):
            #Calcular las distancias del punto pD con los demás datos, y las almacena en dist
            dist = []
            for n in range(len(d)):
                if n == inD:
                    dist.append(444444)
                else:
                    dist.append((pD[0] - d[n][0])**2 + (pD[1] - d[n][1])**2)

            #Obtener las k distancias más cercanas y las clases de estos
            kPuntosMasCerca = np.argsort(dist)[:k]
            kClasesMasCerca = [d[i][2] for i in kPuntosMasCerca]
            count = 0
            if 1 in kClasesMasCerca:
                count += 1

            #Clase con más puntos cercanos
            claseMasCerca = Counter(kClasesMasCerca).most_common(1)[0][0]

            #Si la clase con más puntos cercanos no es la clase del punto pD, este se borra
            clases = [i for i in range(self.numClases)]
            clases.remove(pD[2])
            if any(elem in kClasesMasCerca for elem in clases):#pD[2] in claseMasCerca:
                d.pop(inD)

        return d


    def graficarDatosEntrada(self, atributos, ruta):
        """Grafica los datos de entrada en un plano de dispersión cuyos ejes son los atributos especificados.
        Almacena la gráfica en la ruta introducida como parámetro"""

        #Define el título de la gráfica y el nombre de los ejes
        plt.title("Dispersión de datos de entrada")
        plt.xlabel("Atributo {}".format(atributos[0]))
        plt.ylabel("Atributo {}".format(atributos[1]))
        #plt.axis([160, 180, 220, 230])

        #Recorre la matriz de datos y los grafica por clases
        for cl,dat in enumerate(self.data):
            plt.scatter(dat[:,atributos[0]], dat[:,atributos[1]], label = 'Clase {}'.format(cl))

        #Coloca la legenda de los datos correspondiente a su clase
        plt.legend(loc=3)
        plt.savefig(ruta)

        #Cierra la gráfica
        plt.close()

    def graficarDatos(self, datos, atributos, ruta):
        """Grafica los datos introducidos en un plano de dispersión cuyos ejes son los atributos especificados.
        Almacena la gráfica en la ruta introducida como parámetro"""

        #Define el título de la gráfica y el nombre de los ejes
        plt.title("Dispersión de datos suavizados")
        plt.xlabel("Atributo {}".format(atributos[0]))
        plt.ylabel("Atributo {}".format(atributos[1]))
        #plt.axis([160, 180, 220, 230])

        #Recorre la matriz de datos y los grafica por clases
        datosForma = [np.zeros(len(atributos)) for x in range(self.numClases)]
        for dato in datos:
            clase = dato[2]
            datLinea = [dato[0], dato[1]]
            if not datosForma[clase].any():
                datosForma[clase] = np.array(datLinea)
            else:
                datosForma[clase] = np.vstack([datosForma[clase], datLinea])

        for cl,dat in enumerate(datosForma):
            if len(dat) != 2:
                plt.scatter(dat[:,0], dat[:,1], label = 'Clase {}'.format(cl))
            else:
                plt.scatter(dat[0], dat[1], label = 'Clase {}'.format(cl))

        #Coloca la legenda de los datos correspondiente a su clase
        plt.legend(loc=3)
        plt.savefig(ruta)

        #Cierra la gráfica
        plt.close()
