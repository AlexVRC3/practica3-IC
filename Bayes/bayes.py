import numpy as np

class ClasificadorBayes:
    def __init__(self, datos, etiquetas):
        self.datos_por_clase = {}
        self.centros = {}
        self.matrices_covarianza = {}
        self.inicializa_datos(datos, etiquetas)

    def inicializa_datos(self, datos, etiquetas):
        for punto, clase in zip(datos, etiquetas):
            if clase not in self.datos_por_clase:
                self.datos_por_clase[clase] = []
            self.datos_por_clase[clase].append(punto)
        for clase in self.datos_por_clase:
            self.datos_por_clase[clase] = np.array(self.datos_por_clase[clase])

    def ejecutar(self):
        self.calcular_centros()
        self.calcular_matrices_covarianza()

    def calcular_centros(self):
        for clase, puntos in self.datos_por_clase.items():
            self.centros[clase] = np.mean(puntos, axis=0)

    def calcular_matrices_covarianza(self):
        for clase, puntos in self.datos_por_clase.items():
            centro = self.centros[clase]
            dif = puntos - centro
            matriz = np.dot(dif.T, dif) / len(puntos)
            self.matrices_covarianza[clase] = matriz

    def clasificar(self, nuevo_punto):
        mejor_clase = None
        mejor_valor = -np.inf

        for clase in self.centros:
            centro = self.centros[clase]
            cov = self.matrices_covarianza[clase]
            try:
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
            except np.linalg.LinAlgError:
                continue

            dif = nuevo_punto - centro
            exponente = -0.5 * np.dot(np.dot(dif.T, inv_cov), dif)
            d = len(nuevo_punto)
            coef = 1 / (np.power(2 * np.pi, d / 2) * np.sqrt(det_cov))
            verosimilitud = coef * np.exp(exponente)

            if verosimilitud > mejor_valor:
                mejor_valor = verosimilitud
                mejor_clase = clase

        return mejor_clase


# Funciones auxiliares
def cargar_datos_con_clase(path):
    X, y = [], []
    with open(path, 'r') as f:
        for line in f:
            partes = line.strip().split(',')
            if len(partes) == 5:
                X.append(list(map(float, partes[:4])))
                y.append(partes[4])
    return np.array(X), np.array(y)

def cargar_test_punto(path):
    with open(path, 'r') as f:
        linea = f.readline().strip()
        partes = linea.split(',')
        x = np.array(list(map(float, partes[:4])))
        y = partes[4] if len(partes) > 4 else None
        return x, y


# Ejecución principal
if __name__ == "__main__":
    # Entrenamiento
    X_train, y_train = cargar_datos_con_clase("../Iris2Clases.txt")
    modelo = ClasificadorBayes(X_train, y_train)
    modelo.ejecutar()

    print("\nClasificación de test:")
    for testfile in ["../TestIris01.txt", "../TestIris02.txt", "../TestIris03.txt"]:
        x_test, clase_real = cargar_test_punto(testfile)
        clase_predicha = modelo.clasificar(x_test)
        print(f"{testfile} => pertenece a la clase {clase_predicha}")