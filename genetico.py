import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

# RUTAS
RUTA_DATASET = "iris.data"
RUTA_MODELO  = "modelo.joblib"
RUTA_SALIDA  = "contraejemplos.csv"

# MODELO
TIPO_MODELO  = "mlp" # "svm", "mlp", "perceptron"

# HIPERPARÁMETROS GA
TAMANO_POBLACION = 300
GENERACIONES     = 150
TASA_MUTACION    = 0.2
NUM_PARES        = 70

def cargar_datos(ruta_archivo, columna_objetivo=None):
    df = pd.read_csv(ruta_archivo)
    
    if columna_objetivo is None:
        columna_objetivo = df.columns[-1]
        
    y = df[columna_objetivo]
    X = df.drop(columns=[columna_objetivo])
    
    X = pd.get_dummies(X, drop_first=True)
    
    limites = np.vstack([X.min().values, X.max().values]).T
    
    return X, y, limites, X.columns.tolist()

def entrenar_clasificador(X, y, tipo_modelo="svm"):
    if tipo_modelo == "svm":
        from sklearn.svm import SVC
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    elif tipo_modelo == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=0)
    elif tipo_modelo == "perceptron":
        from sklearn.linear_model import Perceptron
        clf = Perceptron(max_iter=1000, tol=1e-3, random_state=0)
    else:
        raise ValueError(f"Modelo '{tipo_modelo}' no soportado.")
        
    modelo = make_pipeline(StandardScaler(), clf)
    modelo.fit(X, y)
    
    y_pred = modelo.predict(X)
    print(f"Precisión del modelo ({tipo_modelo}) en entrenamiento: {accuracy_score(y, y_pred):.2f}")
    return modelo

def evaluar_poblacion(poblacion, modelo, d_caracteristicas, nombres_caracteristicas, sigma_share):
    puntos_a = poblacion[:, :d_caracteristicas]
    puntos_b = poblacion[:, d_caracteristicas:]
    
    df_a = pd.DataFrame(puntos_a, columns=nombres_caracteristicas)
    df_b = pd.DataFrame(puntos_b, columns=nombres_caracteristicas)
    
    clases_a = modelo.predict(df_a)
    clases_b = modelo.predict(df_b)
    
    distancias = np.linalg.norm(puntos_a - puntos_b, axis=1)
    
    misma_clase = (clases_a == clases_b)
    distancias[misma_clase] += 999999.0
    
    # === FITNESS SHARING POR TRANSICIÓN DE CLASE ===
    # Contamos cuántos individuos están explorando cada frontera específica (ej. Clase A vs Clase B)
    idx_validos = ~misma_clase
    
    transiciones = np.array([f"{a}_{b}" for a, b in zip(clases_a, clases_b)])
    
    if np.any(idx_validos):
        transiciones_validas = transiciones[idx_validos]
        unicas, conteos = np.unique(transiciones_validas, return_counts=True)
        
        # Penalizamos la distancia multiplicándola por la cantidad de clones en esa misma frontera.
        # Esto fuerza al algoritmo a mantener un equilibrio y explorar TODAS las combinaciones de clases.
        for trans, count in zip(unicas, conteos):
            mascara = (transiciones == trans) & idx_validos
            distancias[mascara] *= count

    return distancias

def inicializar_poblacion(tamano_poblacion, limites, d_caracteristicas):
    limites_completos = np.vstack([limites, limites])
    poblacion = np.random.uniform(limites_completos[:, 0], limites_completos[:, 1], size=(tamano_poblacion, 2 * d_caracteristicas))
    return poblacion

def seleccion_torneo(poblacion, aptitudes, k=3):
    indices_seleccion = np.random.randint(0, len(poblacion), size=k)
    mejor_indice = indices_seleccion[np.argmin(aptitudes[indices_seleccion])]
    return poblacion[mejor_indice]

def cruce(padre1, padre2, alfa=0.5):
    hijo1 = alfa * padre1 + (1 - alfa) * padre2
    hijo2 = alfa * padre2 + (1 - alfa) * padre1
    return hijo1, hijo2

def mutar(individuo, limites_completos, tasa_mutacion=0.1, sigma=0.1):
    mutado = np.copy(individuo)
    for i in range(len(mutado)):
        if np.random.rand() < tasa_mutacion:
            ruido = np.random.normal(0, sigma * (limites_completos[i, 1] - limites_completos[i, 0]))
            mutado[i] += ruido
            mutado[i] = np.clip(mutado[i], limites_completos[i, 0], limites_completos[i, 1])
    return mutado

def extraer_mejores_pares_diversos(poblacion, aptitudes, n_pares=40, dist_minima=0.5):
    """
    Filtra la población final para obtener hasta 'n_pares' pares válidos separados 
    por al menos 'dist_minima' a través de toda la frontera hallada.
    """
    indices_validos = np.where(aptitudes < 900000)[0]
    if len(indices_validos) == 0:
        return []
    
    pob_valida = poblacion[indices_validos]
    aptitudes_validas = aptitudes[indices_validos]
    
    orden = np.argsort(aptitudes_validas)
    pob_ordenada = pob_valida[orden]
    
    d_dimensiones = pob_ordenada.shape[1] // 2
    
    pares_seleccionados = []
    
    for individuo in pob_ordenada:
        punto_a = individuo[:d_dimensiones]
        muy_cerca = False
        
        for sel in pares_seleccionados:
            punto_a_sel = sel[:d_dimensiones]
            if np.linalg.norm(punto_a - punto_a_sel) < dist_minima:
                muy_cerca = True
                break
                
        if not muy_cerca:
            pares_seleccionados.append(individuo)
            if len(pares_seleccionados) == n_pares:
                break
                
    return np.array(pares_seleccionados)

def algoritmo_genetico(modelo, limites, nombres_caracteristicas, tamano_poblacion=300, generaciones=150, tasa_mutacion=0.2, num_pares=40):
    d_caracteristicas = limites.shape[0]
    poblacion = inicializar_poblacion(tamano_poblacion, limites, d_caracteristicas)
    limites_completos = np.vstack([limites, limites])
    
    # La distancia elegida a filtrar para limpiar los pares finales
    dist_minima_salida = np.mean(limites[:, 1] - limites[:, 0]) * 0.1
    
    historial_mejores = []
    mejor_individuo_global = None
    mejor_aptitud_global = float('inf')
    
    print("\nIniciando algoritmo genético...\n")
    for gen in range(generaciones):
        aptitudes = evaluar_poblacion(poblacion, modelo, d_caracteristicas, nombres_caracteristicas, sigma_share=1.0)
        
        indice_min_aptitud = np.argmin(aptitudes)
        mejor_aptitud_gen = aptitudes[indice_min_aptitud]
        
        if mejor_aptitud_gen < mejor_aptitud_global:
            mejor_aptitud_global = mejor_aptitud_gen
            mejor_individuo_global = poblacion[indice_min_aptitud].copy()
            
        historial_mejores.append(mejor_aptitud_global)
        
        if gen % 10 == 0 or gen == generaciones - 1:
            valor = mejor_aptitud_global if mejor_aptitud_global < 900000 else "Aún no hay pares válidos"
            print(f"Generación {gen}: Mejor Distancia = {valor if isinstance(valor, str) else round(valor, 4)}")

        nueva_poblacion = []
        if mejor_aptitud_global < 900000:
            nueva_poblacion.extend([mejor_individuo_global, mejor_individuo_global])
            
        while len(nueva_poblacion) < tamano_poblacion:
            p1 = seleccion_torneo(poblacion, aptitudes)
            p2 = seleccion_torneo(poblacion, aptitudes)
            
            h1, h2 = cruce(p1, p2, alfa=np.random.rand())
            h1 = mutar(h1, limites_completos, tasa_mutacion)
            h2 = mutar(h2, limites_completos, tasa_mutacion)
            
            nueva_poblacion.extend([h1, h2])
            
        poblacion = np.array(nueva_poblacion[:tamano_poblacion])
        
    # Re-evaluar cruda para clasificar al final por distancia purista 
    # y así exportar los pares limpios des-ponderados 
    puntos_a = poblacion[:, :d_caracteristicas]
    puntos_b = poblacion[:, d_caracteristicas:]
    df_a = pd.DataFrame(puntos_a, columns=nombres_caracteristicas)
    df_b = pd.DataFrame(puntos_b, columns=nombres_caracteristicas)
    clases_a = modelo.predict(df_a)
    clases_b = modelo.predict(df_b)
    aptitudes_finales = np.linalg.norm(puntos_a - puntos_b, axis=1)
    aptitudes_finales[clases_a == clases_b] += 999999.0
    
    # Extraemos 40 pares super diversos a lo ancho
    pares = extraer_mejores_pares_diversos(poblacion, aptitudes_finales, n_pares=num_pares, dist_minima=dist_minima_salida)
    
    return pares, historial_mejores

def exportar_resultados(pares, modelo, nombres_caracteristicas, archivo_csv="contraejemplos.csv"):
    d_dimension = len(nombres_caracteristicas)
    filas = []
    
    for par in pares:
        punto_a = par[:d_dimension]
        punto_b = par[d_dimension:]
        
        delta = punto_b - punto_a
        cambios = np.abs(delta) > 1e-3
        
        fila = {}
        for i, col in enumerate(nombres_caracteristicas):
            fila[col] = punto_a[i]
            fila[f"ce_{col}"] = punto_b[i]
            fila[f"delta_{col}"] = delta[i]
            fila[f"changed_{col}"] = int(cambios[i])
            
        df_a = pd.DataFrame([punto_a], columns=nombres_caracteristicas)
        fila["pred_orig"] = modelo.predict(df_a)[0]
        
        fila["dist_l2"] = np.linalg.norm(delta)
        fila["num_features_changed"] = np.sum(cambios)
        
        filas.append(fila)
        
    df_resultados = pd.DataFrame(filas)
    df_resultados.to_csv(archivo_csv, index=False)
    print(f"\nExportados {len(pares)} pares a '{archivo_csv}'")

if __name__ == "__main__":
    
    X_entrenamiento, y_entrenamiento, limites_datos, nombres_dim = cargar_datos(RUTA_DATASET)
    
    modelo_entrenado = entrenar_clasificador(X_entrenamiento, y_entrenamiento, TIPO_MODELO)
    
    joblib.dump({"modelo": modelo_entrenado, "nombres": nombres_dim}, RUTA_MODELO)
    
    # Ejecutamos el algoritmo genético con los hiperparámetros globales
    pares_finales, historial = algoritmo_genetico(
        modelo=modelo_entrenado, 
        limites=limites_datos,
        nombres_caracteristicas=nombres_dim,
        tamano_poblacion=TAMANO_POBLACION, 
        generaciones=GENERACIONES,
        tasa_mutacion=TASA_MUTACION,
        num_pares=NUM_PARES
    )
    
    if len(pares_finales) > 0:
        exportar_resultados(pares_finales, modelo_entrenado, nombres_dim, RUTA_SALIDA)
    else:
        print("\nNo se encontró ningún par válido.")
