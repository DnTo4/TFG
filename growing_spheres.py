import numpy as np

def vectores_unitarios(n, d, rng):
    # Genera n vectores aleatorios en R^d
    vec = rng.normal(size=(n, d))

    # Normaliza a longitud 1 (vectores unitarios)
    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12) # el +1e-12 evita divisiones por 0
    return vec

# Muestrea n puntos en la cascara entre radios r_inf y r_sup alrededor de centro
def muestrea_cascara(centro, r_inf, r_sup, n, rng):
    # Dimensiones de los puntos
    d = centro.size

    # Genera n radios aleatorios entre r_inf y r_sup
    radios = rng.uniform(r_inf, r_sup, size=n).reshape(-1, 1)

    # Genera n vectores unitarios aleatorios y los escala por los radios
    return centro + vectores_unitarios(n, d, rng) * radios

# Growing Spheres 
def growing_spheres_generacion(predict_fn, x, *, muestras=512, ancho_banda=0.5, max_iters=200, random_state=0):
    # Reducir dimensiones de entrada y guardar prediccion original
    x = np.asarray(x, float).ravel()
    rng = np.random.default_rng(random_state)
    y_x = predict_fn(x.reshape(1, -1))[0]

    # Crea muestras candidatos en una esfera de radio eta centrada en x
    eta = float(ancho_banda)
    iter = 0
    cand = muestrea_cascara(x, 0.0, eta, muestras, rng)

    # Si hay de otra clase, reducimos eta hasta encontrar una banda sin
    while np.any(predict_fn(cand) != y_x) and iter < max_iters:
        eta *= 0.5
        cand = muestrea_cascara(x, 0.0, eta, muestras, rng)
        iter += 1

    # Define banda [a0 = eta, a1 = 2 * eta] y muestrea en ella
    a0, a1 = eta, 2 * eta
    iter = 0
    cand = muestrea_cascara(x, a0, a1, muestras, rng)

    # Mientras no haya de otra clase, aumenta la banda en eta
    while not np.any(predict_fn(cand) != y_x) and iter < max_iters:
        a0 = a1
        a1 = a1 + eta
        cand = muestrea_cascara(x, a0, a1, muestras, rng)
        iter += 1

    # Elegir el candidato mas cercano en la ultima banda
    labels = predict_fn(cand)
    idx = np.where(labels != y_x)[0]

    # Si no hay ninguno, lanzar error
    if idx.size == 0:
        raise RuntimeError("No se encontro contraejemplo")
    
    # Seleccionar el mas cercano por distancia euclidea
    i = idx[np.argmin(np.linalg.norm(cand[idx] - x, axis=1))]

    return cand[i]

# Reduce el numero de variables modificadas en el contraejemplo
def feature_selection(predict_fn, x, CEj):
    # Asegurarse de que x y CEj son arrays 1D
    x = np.asarray(x, float).ravel()
    cEj = np.asarray(CEj, float).ravel().copy()

    # Obtener la prediccion original
    y_x = predict_fn(x.reshape(1, -1))[0]

    # intentar apagar la caracteristica con menor cambio manteniendo que siga siendo de diferente clase
    while predict_fn(cEj.reshape(1, -1))[0] != y_x:
        # Cacula el cambio
        dif = np.abs(cEj - x)

        #Ignora las iguales
        dif[dif == 0.0] = np.inf

        # Eloge la variable de cambio minimo
        k = int(np.argmin(dif))

        # Intentar devolver esa variable a su valor original
        prueba = cEj.copy()
        prueba[k] = x[k]

        # Si sigue siendo de la otra clase, aceptar el cambio
        if predict_fn(prueba.reshape(1, -1))[0] != y_x:
            cEj = prueba
        else:
            break

    return cEj