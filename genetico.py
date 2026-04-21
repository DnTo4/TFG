import numpy as np
import pandas as pd
import joblib

from perceptron import train_model as train_pt_model
from svm import train_model as train_svm_model
from mlp import train_model as train_mlp_model

# RUTAS
RUTA_DATASET_TRAIN = "diabetes.csv"
RUTA_DATASET_TEST  = "diabetes.csv"
RUTA_MODELO  = "modelo.joblib"
RUTA_SALIDA  = "contraejemplos.csv"

# MODELO
TIPO_MODELO  = "mlp" # "svm", "mlp", "perceptron"

# HIPERPARÁMETROS GA
TAMANO_POBLACION = 500   
GENERACIONES     = 300   
TASA_MUTACION    = 0.25  
NUM_PARES        = 50    

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
    
    # Calcular límites estandarizados
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_combined = pd.concat([X_train, X_test], axis=0)
    scaler.fit(X_combined)
    
    X_scaled = scaler.transform(X_combined)
    limites_scaled = np.vstack([X_scaled.min(axis=0), X_scaled.max(axis=0)]).T
    
    return modelo, limites_scaled, nombres, scaler

def evaluar_poblacion(poblacion, modelo, d_caracteristicas, nombres_caracteristicas, sigma_share, scaler):
    # Divide los individuos usando el espacio estandarizado (para calcular bien las distancias)
    puntos_a_scaled = poblacion[:, :d_caracteristicas]
    puntos_b_scaled = poblacion[:, d_caracteristicas:]
    
    # Desescala a valores originales temporalmente para que el modelo predictivo funcione bien
    puntos_a_raw = scaler.inverse_transform(puntos_a_scaled)
    puntos_b_raw = scaler.inverse_transform(puntos_b_scaled)
    
    df_a = pd.DataFrame(puntos_a_raw, columns=nombres_caracteristicas)
    df_b = pd.DataFrame(puntos_b_raw, columns=nombres_caracteristicas)
    
    # Predice las clases usando valores reales
    clases_a = modelo.predict(df_a)
    clases_b = modelo.predict(df_b)
    
    # Calcula la distancia euclidiana en el ENTORNO ESTANDARIZADO
    distancias = np.linalg.norm(puntos_a_scaled - puntos_b_scaled, axis=1)
    
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

def algoritmo_genetico(modelo, limites, nombres_caracteristicas, scaler, tamano_poblacion=300, generaciones=150, tasa_mutacion=0.2, num_pares=40):
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
        aptitudes = evaluar_poblacion(poblacion, modelo, d_caracteristicas, nombres_caracteristicas, sigma_share=1.0, scaler=scaler)
        
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
    puntos_a_scaled = poblacion[:, :d_caracteristicas]
    puntos_b_scaled = poblacion[:, d_caracteristicas:]
    df_a = pd.DataFrame(scaler.inverse_transform(puntos_a_scaled), columns=nombres_caracteristicas)
    df_b = pd.DataFrame(scaler.inverse_transform(puntos_b_scaled), columns=nombres_caracteristicas)
    clases_a = modelo.predict(df_a)
    clases_b = modelo.predict(df_b)
    
    aptitudes_finales = np.linalg.norm(puntos_a_scaled - puntos_b_scaled, axis=1)
    aptitudes_finales[clases_a == clases_b] += 999999.0
    
    # Extraer n_pares diversos en el espacio estandarizado
    pares_scaled = filtrar_contraejemplos(poblacion, aptitudes_finales, n_pares=num_pares, dist_minima=dist_minima_salida)
    
    # Devolver los pares desescalados a sus valores reales originales para exportar correctamente
    pares_raw = []
    for par in pares_scaled:
        p_a = scaler.inverse_transform([par[:d_caracteristicas]])[0]
        p_b = scaler.inverse_transform([par[d_caracteristicas:]])[0]
        pares_raw.append(np.concatenate([p_a, p_b]))
    
    return np.array(pares_raw), historial_mejores

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
    
    modelo_entrenado, limites_datos, nombres_dim, scaler_genetico = entrenar_clasificador(RUTA_DATASET_TRAIN, RUTA_DATASET_TEST, TIPO_MODELO)
    
    joblib.dump({"modelo": modelo_entrenado, "nombres": nombres_dim}, RUTA_MODELO)
    
    # Ejecutar el algoritmo genético con los hiperparámetros globales
    pares_finales, historial = algoritmo_genetico(
        modelo=modelo_entrenado, 
        limites=limites_datos,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler_genetico,
        tamano_poblacion=TAMANO_POBLACION, 
        generaciones=GENERACIONES,
        tasa_mutacion=TASA_MUTACION,
        num_pares=NUM_PARES
    )
    
    if len(pares_finales) > 0:
        exportar_resultados(pares_finales, modelo_entrenado, nombres_dim, RUTA_SALIDA)
    else:
        print("\nNo se encontró ningún par válido.")
