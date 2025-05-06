import numpy as np

class Lloyd:
    def __init__(self, puntos, centros, tolerancia=1e-10, num_max_iter=10, r_aprendizaje=0.1):
        self.puntos = [np.array(p) for p in puntos]
        self.centros = [np.array(c) for c in centros]
        self.centros_ant = []
        self.tolerancia = tolerancia
        self.num_max_iter = num_max_iter
        self.r_aprendizaje = r_aprendizaje

    def execute(self):
        num_iter = 0
        while num_iter < self.num_max_iter:
            num_iter += 1
            self.centros_ant = [np.copy(c) for c in self.centros]
            actualiza = 0
            for i, punto in enumerate(self.puntos):
                indice_mejor = self.competicion(punto)
                self.actualiza_centro(indice_mejor, i)
            
            if self.fin():
                break

    def fin(self):
        for c, c_ant in zip(self.centros, self.centros_ant):
            if self.distancia(c, c_ant) >= self.tolerancia:
                return False
        return True

    def competicion(self, punto):
        distancias = [self.distancia(punto, centro) for centro in self.centros]
        return np.argmin(distancias)

    def actualiza_centro(self, i_centro, i_punto):
        self.centros[i_centro] += self.r_aprendizaje * (self.puntos[i_punto] - self.centros[i_centro])

    def distancia(self, punto, centro):
        return np.linalg.norm(punto - centro)

    def clasificar_nuevo(self, punto):
        return self.competicion(punto)

    def get_centros(self):
        return [c.tolist() for c in self.centros]

    def set_centros(self, centros):
        self.centros = [np.array(c) for c in centros]
        self.centros_ant = []


def cargar_datos(ruta_archivo):
    puntos = []
    with open(ruta_archivo, 'r') as f:
        for line in f:
            if line.strip():
                partes = line.strip().split(",")
                puntos.append(list(map(float, partes[:4])))  # solo atributos numéricos
    return puntos

def cargar_test_punto(ruta_archivo):
    with open(ruta_archivo, 'r') as f:
        line = f.readline().strip()
        partes = line.split(",")
        return list(map(float, partes[:4]))  # solo convierte los primeros 4 atributos a float


if __name__ == "__main__":
    # Cargar los datos de entrenamiento desde un archivo
    puntos = cargar_datos("../Iris2Clases.txt")
    k = 2  # porque Iris2Clases.txt tiene 2 clases
    centros_iniciales = puntos[:k]  # Se toman los primeros k puntos como centros iniciales

    # Crear y entrenar el modelo de Lloyd
    modelo = Lloyd(puntos, centros_iniciales)
    modelo.execute()

    print("Centros finales:")
    for c in modelo.get_centros():
        print(c)

    # Clasificación de nuevos puntos de test
    print("\nClasificación de test:")
    test_files = ["../TestIris01.txt", "../TestIris02.txt", "../TestIris03.txt"]
    for testfile in test_files:
        punto = cargar_test_punto(testfile)
        cluster = modelo.clasificar_nuevo(punto)
        print(f"{testfile} => pertenece al cluster {cluster}")
