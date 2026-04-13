import numpy as np
import pandas as pd
import joblib

from perceptron import train_model as train_pt_model
from svm import train_model as train_svm_model
from mlp import train_model as train_mlp_model

# RUTAS
RUTA_DATASET_TRAIN = "iris.data"
RUTA_DATASET_TEST  = "iris.data"
RUTA_MODELO  = "modelo.joblib"
RUTA_SALIDA  = "contraejemplos.csv"

# MODELO
TIPO_MODELO  = "mlp" # "svm", "mlp", "perceptron"

# HIPERPARÁMETROS GA
TAMANO_POBLACION = 500   # (Antes 300) Más población = mejor cobertura y diversidad en el espacio
GENERACIONES     = 300   # (Antes 150) Más tiempo para refinar los puntos y acercarlos a la frontera
TASA_MUTACION    = 0.25  # (Antes 0.2) Ligeramente superior para evitar estancamiento prematuro
NUM_PARES        = 50    # (Antes 70) Menos pares requeridos aísla a que solo se seleccionen los de mayor calidad sin chocar con la distancia mínima

def entrenar_clasificador(ruta_train, ruta_test, tipo_modelo="svm"):
    modelos = {
        "perceptron": train_pt_model,
        "svm": train_svm_model,
        "mlp": train_mlp_model
    }
    if tipo_modelo not in modelos:
        raise ValueError(f"Modelo '{tipo_modelo}' no soportado.")
        
    entrenar_func = modelos[tipo_modelo]
    modelo, (X_train, y_train, X_test, y_test), acc, nombres = entrenar_func(ruta_train, ruta_test, None)
    
    print(f"Precisión del modelo ({tipo_modelo}) en prueba: {acc:.2f}")
    
    # Calcular límites
    X_combined = pd.concat([X_train, X_test], axis=0)
    limites = np.vstack([X_combined.min().values, X_combined.max().values]).T
    
    return modelo, limites, nombres

def evaluar_poblacion(poblacion, modelo, d_caracteristicas, nombres_caracteristicas, sigma_share):
    # Divide los individuos en puntos originales y contraejemplos
    puntos_a = poblacion[:, :d_caracteristicas]
    puntos_b = poblacion[:, d_caracteristicas:]
    
    # Convierte los puntos a DataFrames
    df_a = pd.DataFrame(puntos_a, columns=nombres_caracteristicas)
    df_b = pd.DataFrame(puntos_b, columns=nombres_caracteristicas)
    
    # Predice las clases
    clases_a = modelo.predict(df_a)
    clases_b = modelo.predict(df_b)
    
    # Calcula la distancia euclidiana entre los puntos
    distancias = np.linalg.norm(puntos_a - puntos_b, axis=1)
    
    # Penaliza los individuos que están en la misma clase
    misma_clase = (clases_a == clases_b)
    distancias[misma_clase] += 999999.0
    

    # Contamos cuántos individuos están explorando cada frontera específica (Fitness sharing)
    idx_validos = ~misma_clase
    
    # Identifica las transiciones de clase (0_1, 1_2, ...)
    transiciones = np.array([f"{a}_{b}" for a, b in zip(clases_a, clases_b)])
    
    if np.any(idx_validos):
        transiciones_validas = transiciones[idx_validos]
        unicas, conteos = np.unique(transiciones_validas, return_counts=True)
        
        # Penalizar la distancia multiplicándola por la cantidad de clones en esa misma frontera.
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
    # Cruce aritmético (trazar una línea entre los padres y tomar puntos intermedios)
    hijo1 = alfa * padre1 + (1 - alfa) * padre2
    hijo2 = alfa * padre2 + (1 - alfa) * padre1
    return hijo1, hijo2

def mutar(individuo, limites_completos, tasa_mutacion=0.1, sigma=0.1):
    # Añadir ruido gaussiano 
    mutado = np.copy(individuo)
    for i in range(len(mutado)):
        if np.random.rand() < tasa_mutacion:
            ruido = np.random.normal(0, sigma * (limites_completos[i, 1] - limites_completos[i, 0]))
            mutado[i] += ruido
            # Mantener los valores dentro de los límites
            mutado[i] = np.clip(mutado[i], limites_completos[i, 0], limites_completos[i, 1])
    return mutado

def filtrar_contraejemplos(poblacion, aptitudes, n_pares=40, dist_minima=0.5):
    # Elimina los individuos que están en la misma clase
    indices_validos = np.where(aptitudes < 900000)[0]
    if len(indices_validos) == 0:
        return []
    
    pob_valida = poblacion[indices_validos]
    aptitudes_validas = aptitudes[indices_validos]
    
    # Ordena los individuos por aptitud
    orden = np.argsort(aptitudes_validas)
    pob_ordenada = pob_valida[orden]
    
    # Obtiene las dimensiones de cada punto
    d_dimensiones = pob_ordenada.shape[1] // 2
    
    pares_seleccionados = []
    
    for individuo in pob_ordenada:
        punto_orig = individuo[:d_dimensiones]
        cerca = False
        
        for sel in pares_seleccionados:
            # Calcula la distancia entre el punto actual y los puntos ya seleccionados
            punto_guardado = sel[:d_dimensiones]
            # Si la distancia es menor a la distancia mínima, se descarta el punto
            if np.linalg.norm(punto_orig - punto_guardado) < dist_minima:
                cerca = True
                break
                
        if not cerca:
            pares_seleccionados.append(individuo)
            if len(pares_seleccionados) == n_pares:
                break
                
    return np.array(pares_seleccionados)

def algoritmo_genetico(modelo, limites, nombres_caracteristicas, tamano_poblacion=300, generaciones=150, tasa_mutacion=0.2, num_pares=40):
    # Obtiene las dimensiones de cada punto
    d_caracteristicas = limites.shape[0]

    # Inicializa la población
    poblacion = inicializar_poblacion(tamano_poblacion, limites, d_caracteristicas)

    # Calcula los límites 
    limites_completos = np.vstack([limites, limites])
    
    # La distancia elegida a filtrar para limpiar los pares finales
    dist_minima_salida = np.mean(limites[:, 1] - limites[:, 0]) * 0.1
    
    historial_mejores = []
    elite = None
    mejor_aptitud = float('inf')
    
    print("\nIniciando algoritmo genético...\n")
    for gen in range(generaciones):
        # Evalúa la aptitud de cada individuo
        aptitudes = evaluar_poblacion(poblacion, modelo, d_caracteristicas, nombres_caracteristicas, sigma_share=1.0)
        
        # Obtiene el índice del individuo con la mejor aptitud
        indice_min_aptitud = np.argmin(aptitudes)
        mejor_aptitud_gen = aptitudes[indice_min_aptitud]
        
        # Si la aptitud del individuo actual es mejor que la mejor aptitud global, se actualiza
        if mejor_aptitud_gen < mejor_aptitud:
            mejor_aptitud = mejor_aptitud_gen
            elite = poblacion[indice_min_aptitud].copy()
            
        historial_mejores.append(mejor_aptitud)
        
        # Imprime la mejor aptitud cada 10 generaciones
        if gen % 10 == 0 or gen == generaciones - 1:
            valor = mejor_aptitud if mejor_aptitud < 900000 else "Aún no hay pares válidos"
            print(f"Generación {gen}: Mejor Distancia = {valor if isinstance(valor, str) else round(valor, 4)}")

        # Crea una nueva población
        nueva_poblacion = []
        # Si hay un individuo con buena aptitud, se añade a la nueva población
        if mejor_aptitud < 900000:
            nueva_poblacion.extend([elite, elite])
            
        # Mientras la nueva población no esté completa, se añaden individuos
        while len(nueva_poblacion) < tamano_poblacion:
            p1 = seleccion_torneo(poblacion, aptitudes)
            p2 = seleccion_torneo(poblacion, aptitudes)
            
            h1, h2 = cruce(p1, p2, alfa=np.random.rand())
            h1 = mutar(h1, limites_completos, tasa_mutacion)
            h2 = mutar(h2, limites_completos, tasa_mutacion)
            
            nueva_poblacion.extend([h1, h2])
            
        poblacion = np.array(nueva_poblacion[:tamano_poblacion])
        
    # Re-evaluar  
    puntos_a = poblacion[:, :d_caracteristicas]
    puntos_b = poblacion[:, d_caracteristicas:]
    df_a = pd.DataFrame(puntos_a, columns=nombres_caracteristicas)
    df_b = pd.DataFrame(puntos_b, columns=nombres_caracteristicas)
    clases_a = modelo.predict(df_a)
    clases_b = modelo.predict(df_b)
    aptitudes_finales = np.linalg.norm(puntos_a - puntos_b, axis=1)
    aptitudes_finales[clases_a == clases_b] += 999999.0
    
    # Extraer n_pares diversos
    pares = filtrar_contraejemplos(poblacion, aptitudes_finales, n_pares=num_pares, dist_minima=dist_minima_salida)
    
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
    
    modelo_entrenado, limites_datos, nombres_dim = entrenar_clasificador(RUTA_DATASET_TRAIN, RUTA_DATASET_TEST, TIPO_MODELO)
    
    joblib.dump({"modelo": modelo_entrenado, "nombres": nombres_dim}, RUTA_MODELO)
    
    # Ejecutar el algoritmo genético con los hiperparámetros globales
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
