import numpy as np
from collections import defaultdict

class KMeansDifuso:
    def __init__(self, puntos, centros, tolerancia=0.01, b=2):
        self.puntos = [np.array(p) for p in puntos]
        self.centros = [np.array(c) for c in centros]
        self.centros_ant = []
        self.tolerancia = tolerancia
        self.b = b
        self.U = None

    def execute(self):
        while True:
            self.centros_ant = self.centros.copy()
            self.calcula_matriz_pertinencia()
            self.actualiza_centros()
            if self.termino():
                break

    def calcula_matriz_pertinencia(self):
        n_centros = len(self.centros)
        n_puntos = len(self.puntos)
        self.U = np.zeros((n_centros, n_puntos))
        for i in range(n_centros):
            for j in range(n_puntos):
                self.U[i][j] = self.calcula_prob_pertinencia(i, j)

    def calcula_prob_pertinencia(self, i_clase, j_punto):
        dist_ij = self.distancia(self.centros[i_clase], self.puntos[j_punto])
        if dist_ij == 0:
            return 1.0
        num = (1.0 / dist_ij) ** (1 / (self.b - 1))
        denom = sum((1.0 / self.distancia(c, self.puntos[j_punto]) ** (1 / (self.b - 1)) if self.distancia(c, self.puntos[j_punto]) != 0 else float('inf')) for c in self.centros)
        return num / denom

    def actualiza_centros(self):
        for i in range(len(self.centros)):
            self.centros[i] = self.calcula_centro(i)

    def calcula_centro(self, i):
        num = np.zeros_like(self.centros[i])
        denom = 0.0
        for j in range(len(self.puntos)):
            s = self.U[i][j] ** self.b
            num += s * self.puntos[j]
            denom += s
        return num / denom if denom != 0 else num

    def termino(self):
        for i in range(len(self.centros)):
            if self.distancia(self.centros[i], self.centros_ant[i]) >= self.tolerancia:
                return False
        return True

    def distancia(self, a, b):
        return np.linalg.norm(a - b)

    def clasificar_nuevo(self, punto):
        punto = np.array(punto)
        distancias = [self.distancia(punto, c) for c in self.centros]
        return np.argmin(distancias)

    def get_centros(self):
        return [c.tolist() for c in self.centros]

    def get_U(self):
        return self.U


def cargar_datos_con_clase(ruta_archivo):
    puntos = []
    etiquetas = []
    with open(ruta_archivo, 'r') as f:
        for line in f:
            if line.strip():
                partes = line.strip().split(",")
                puntos.append(list(map(float, partes[:4])))
                etiquetas.append(partes[4])
    return puntos, etiquetas

def cargar_test_punto(ruta_archivo):
    with open(ruta_archivo, 'r') as f:
        line = f.readline().strip()
        partes = line.split(",")
        return list(map(float, partes[:4]))

def mapear_clusters_a_clases(modelo, etiquetas):
    asignaciones = [np.argmax([modelo.U[i][j] for i in range(len(modelo.centros))]) for j in range(len(modelo.puntos))]
    conteo = defaultdict(lambda: defaultdict(int))
    for idx, cluster in enumerate(asignaciones):
        clase = etiquetas[idx]
        conteo[cluster][clase] += 1
    print("\nMapeo de clusters a clases (K-Means Difuso):")
    for cluster, clases in conteo.items():
        print(f"Cluster {cluster}:")
        for clase, cuenta in clases.items():
            print(f"  {clase}: {cuenta} ejemplos")

if __name__ == "__main__":
    puntos, etiquetas = cargar_datos_con_clase("../Iris2Clases.txt")
    k = 2
    centros_iniciales = puntos[:k]
    modelo = KMeansDifuso(puntos, centros_iniciales, tolerancia=0.01, b=2)
    modelo.execute()

    print("Centros finales:")
    for c in modelo.get_centros():
        print(c)

    mapear_clusters_a_clases(modelo, etiquetas)

    print("\nClasificación de test:")
    for testfile in ["../TestIris01.txt", "../TestIris02.txt", "../TestIris03.txt"]:
        punto = cargar_test_punto(testfile)
        cluster = modelo.clasificar_nuevo(punto)
        print(f"{testfile} => pertenece al cluster {cluster}")
