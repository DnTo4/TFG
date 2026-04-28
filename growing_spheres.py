import numpy as np
import pandas as pd

def vectores_unitarios(n, d, rng):
    """
    Genera una matriz de n vectores unitarios aleatorios en un espacio de d dimensiones.

    Utiliza una distribución normal para garantizar una dirección uniforme en la hiperesfera
    y luego normaliza cada vector a longitud 1.

    Args:
        n (int): Número de vectores a generar.
        d (int): Número de dimensiones de cada vector.
        rng (np.random.Generator): Generador de números aleatorios para reproducibilidad.

    Returns:
        np.ndarray: Matriz de forma (n, d) con vectores de norma L2 igual a 1.
    """
    # Genera n vectores aleatorios en R^d
    vec = rng.normal(size=(n, d))

    # Normaliza a longitud 1 (vectores unitarios)
    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12) # el +1e-12 evita divisiones por 0
    return vec

def muestrea_cascara(centro, r_inf, r_sup, n, rng):
    """
    Muestrea puntos uniformemente en el espacio comprendido entre dos radios.

    Args:
        centro (np.ndarray): Punto central (coordenadas del dato original).
        r_inf (float): Radio inferior del límite de muestreo.
        r_sup (float): Radio superior del límite de muestreo.
        n (int): Cantidad de puntos a muestrear.
        rng (np.random.Generator): Generador de números aleatorios.

    Returns:
        np.ndarray: Matriz de n puntos distribuidos aleatoriamente en la cáscara definida.
    """
    # Dimensiones de los puntos
    d = centro.size

    # Genera n radios aleatorios entre r_inf y r_sup
    radios = rng.uniform(r_inf, r_sup, size=n).reshape(-1, 1)

    # Genera n vectores unitarios aleatorios y los escala por los radios
    return centro + vectores_unitarios(n, d, rng) * radios

def growing_spheres_generacion(predict_fn, x, *, muestras=512, ancho_banda=0.5, max_iters=200, random_state=0):
    """
    Algoritmo Growing Spheres para encontrar el contraejemplo más cercano.

    El proceso consta de dos fases:
    1. Fase de contracción: Encuentra un radio 'eta' donde no hay enemigos cerca de x.
    2. Fase de crecimiento: Aumenta el radio en capas concéntricas hasta encontrar la frontera de decisión.

    Args:
        predict_fn (callable): Función de predicción del modelo.
        x (pd.DataFrame o np.ndarray): Instancia original para la cual buscar el contraejemplo.
        muestras (int): Número de puntos a generar en cada iteración de muestreo.
        ancho_banda (float): Paso inicial para el radio de las esferas.
        max_iters (int): Límite máximo de iteraciones para evitar bucles infinitos.
        random_state (int): Semilla para el generador de números aleatorios.

    Returns:
        np.ndarray: El contraejemplo (punto con diferente clase) más cercano a x encontrado.

    Raises:
        RuntimeError: Si no se encuentra ningún contraejemplo tras agotar las iteraciones.
    """
    # Guardar nombres de columnas si x es un DataFrame
    columnas = list(x.columns) if hasattr(x, 'columns') else None

    def predict_wrapped(arr):
        if columnas is not None:
            return predict_fn(pd.DataFrame(arr, columns=columnas))
        return predict_fn(arr)

    # Reducir dimensiones de entrada y guardar prediccion original
    x = np.asarray(x, float).ravel()
    rng = np.random.default_rng(random_state)
    y_x = predict_wrapped(x.reshape(1, -1))[0]

    # Crea muestras candidatos en una esfera de radio eta centrada en x
    eta = float(ancho_banda)
    iter = 0
    cand = muestrea_cascara(x, 0.0, eta, muestras, rng)

    # Si hay de otra clase, reducimos eta hasta encontrar una banda sin
    while np.any(predict_wrapped(cand) != y_x) and iter < max_iters:
        eta *= 0.5
        cand = muestrea_cascara(x, 0.0, eta, muestras, rng)
        iter += 1

    # Define banda [a0 = eta, a1 = 2 * eta] y muestrea en ella
    a0, a1 = eta, 2 * eta
    iter = 0
    cand = muestrea_cascara(x, a0, a1, muestras, rng)

    # Mientras no haya de otra clase, aumenta la banda en eta
    while not np.any(predict_wrapped(cand) != y_x) and iter < max_iters:
        a0 = a1
        a1 = a1 + eta
        cand = muestrea_cascara(x, a0, a1, muestras, rng)
        iter += 1

    # Elegir el candidato mas cercano en la ultima banda
    labels = predict_wrapped(cand)
    idx = np.where(labels != y_x)[0]

    # Si no hay ninguno, lanzar error
    if idx.size == 0:
        raise RuntimeError("No se encontro contraejemplo")
    
    # Seleccionar el mas cercano por distancia euclidea
    i = idx[np.argmin(np.linalg.norm(cand[idx] - x, axis=1))]

    return cand[i]

def feature_selection(predict_fn, x, CEj):
    """
    Optimiza el contraejemplo reduciendo el número de características modificadas.

    Iterativamente intenta revertir cada característica del contraejemplo a su valor original.
    Si al revertir una característica el punto sigue perteneciendo a una clase distinta a la original, 
    el cambio se mantiene.

    Args:
        predict_fn (callable): Función de predicción del modelo.
        x (pd.DataFrame o np.ndarray): Instancia original.
        CEj (np.ndarray): Contraejemplo inicial encontrado por Growing Spheres.

    Returns:
        np.ndarray: Un contraejemplo optimizado que difiere de x en el menor número de variables posible.
    """
    columnas = list(x.columns) if hasattr(x, 'columns') else None

    def predict_wrapped(arr):
        if columnas is not None:
            return predict_fn(pd.DataFrame(arr, columns=columnas))
        return predict_fn(arr)

    # Asegurarse de que x y CEj son arrays 1D
    x = np.asarray(x, float).ravel()
    cEj = np.asarray(CEj, float).ravel().copy()

    # Obtener la prediccion original
    y_x = predict_wrapped(x.reshape(1, -1))[0]

    # intentar apagar la caracteristica con menor cambio
    while predict_wrapped(cEj.reshape(1, -1))[0] != y_x:
        # Cacula el cambio
        dif = np.abs(cEj - x)

        #Ignora las iguales
        dif[dif == 0.0] = np.inf

        # Elige la variable de cambio minimo
        k = int(np.argmin(dif))

        # Intentar devolver esa variable a su valor original
        prueba = cEj.copy()
        prueba[k] = x[k]

        # Si sigue siendo de la otra clase, aceptar el cambio
        if predict_wrapped(prueba.reshape(1, -1))[0] != y_x:
            cEj = prueba
        else:
            break

    return cEj
