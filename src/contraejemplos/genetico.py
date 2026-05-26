import numpy as np
import pandas as pd
import joblib

from src.modelos.arbol_decision import train_model as train_arbol_model
from src.modelos.svm import train_model as train_svm_model
from src.modelos.mlp import train_model as train_mlp_model

"""Generación de contraejemplos mediante un algoritmo genético.

Implementa operadores evolutivos específicos (selección por torneo, cruce aritmético,
mutación por ruido gaussiano) y fitness sharing para encontrar contraejemplos 
óptimos y diversos sobre la frontera de decisión.
"""

def entrenar_clasificador(ruta_train, ruta_test, tipo_modelo="svm"):
    """Entrenar el clasificador base y estimar los límites del espacio.

    Carga y entrena el modelo indicado y calcula los límites de las variables 
    mediante StandardScaler.
    """
    modelos = {
        "arbol_decision": train_arbol_model,
        "svm": train_svm_model,
        "mlp": train_mlp_model
    }
    
    if tipo_modelo not in modelos:
        raise ValueError(f"Modelo '{tipo_modelo}' no soportado.")
        
    entrenar_func = modelos[tipo_modelo]
    modelo, (X_train, y_train, X_test, y_test), acc, nombres = entrenar_func(ruta_train, ruta_test, None)
    
    print(f"Precisión del modelo ({tipo_modelo}) en prueba: {acc:.2f}")
    
    # Calcular límites en base a la unión de conjuntos
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_combined = pd.concat([X_train, X_test], axis=0)
    scaler.fit(X_combined)
    
    X_scaled = scaler.transform(X_combined)
    limites_scaled = np.vstack([X_scaled.min(axis=0), X_scaled.max(axis=0)]).T
    
    return modelo, limites_scaled, nombres, scaler

def evaluar_poblacion(poblacion, modelo, d_caracteristicas, nombres_caracteristicas, sigma_share, scaler):
    """Evaluar el fitness de cada individuo.

    Desescala los puntos para evaluar su predicción y computa
    la distancia euclidiana entre ellos, penalizando los pares
    que pertenecen a la misma clase.
    """
    # Separar los dos puntos de cada individuo
    puntos_a_scaled = poblacion[:, :d_caracteristicas]
    puntos_b_scaled = poblacion[:, d_caracteristicas:]
    
    # Desescalar a los valores reales originales
    puntos_a_raw = scaler.inverse_transform(puntos_a_scaled)
    puntos_b_raw = scaler.inverse_transform(puntos_b_scaled)
    
    df_a = pd.DataFrame(puntos_a_raw, columns=nombres_caracteristicas)
    uint_b = pd.DataFrame(puntos_b_raw, columns=nombres_caracteristicas)
    
    # Estimar las clases
    clases_a = modelo.predict(df_a)
    clases_b = modelo.predict(uint_b)
    
    # Distancia euclidiana
    distancias = np.linalg.norm(puntos_a_scaled - puntos_b_scaled, axis=1)
    
    # Penalizar los pares que no cambian de clase
    misma_clase = (clases_a == clases_b)
    distancias[misma_clase] += 999999.0
    
    # Aplicar fitness sharing
    idx_validos = ~misma_clase
    
    # Identificar y agrupar transiciones
    transiciones = np.array([f"{a}_{b}" for a, b in zip(clases_a, clases_b)])
    
    if np.any(idx_validos):
        transiciones_validas = transiciones[idx_validos]
        unicas, conteos = np.unique(transiciones_validas, return_counts=True)
        
        # Penalizar la distancia multiplicando por la densidad en la misma frontera
        for trans, count in zip(unicas, conteos):
            mascara = (transiciones == trans) & idx_validos
            distancias[mascara] *= count

    return distancias

def inicializar_poblacion(tamano_poblacion, limites, d_caracteristicas):
    """Generar una población inicial muestreada uniformemente.

    Muestrea de forma aleatoria y uniforme dentro de los límites
    de cada característica.
    """
    limites_completos = np.vstack([limites, limites])
    poblacion = np.random.uniform(limites_completos[:, 0], limites_completos[:, 1], size=(tamano_poblacion, 2 * d_caracteristicas))
    return poblacion

def seleccion_torneo(poblacion, aptitudes, k=3):
    """Seleccionar un individuo de la población mediante torneo.

    Muestrea un grupo aleatorio de tamaño k y selecciona al individuo
    con mejor fitness.
    """
    indices_seleccion = np.random.randint(0, len(poblacion), size=k)
    mejor_indice = indices_seleccion[np.argmin(aptitudes[indices_seleccion])]
    return poblacion[mejor_indice]

def cruce(padre1, padre2, alfa=0.5):
    """Realizar cruce aritmético entre dos individuos para generar dos descendientes.

    Combina linealmente los genes de los padres en base al factor alfa.
    """
    hijo1 = alfa * padre1 + (1 - alfa) * padre2
    hijo2 = alfa * padre2 + (1 - alfa) * padre1
    return hijo1, hijo2

def mutar(individuo, limites_completos, tasa_mutacion=0.1, sigma=0.1):
    """Aplicar mutación mediante ruido gaussiano.

    Añade perturbaciones normales y acota el gen resultante para respetar
    los límites físicos.
    """
    mutado = np.copy(individuo)
    for i in range(len(mutado)):
        if np.random.rand() < tasa_mutacion:
            ruido = np.random.normal(0, sigma * (limites_completos[i, 1] - limites_completos[i, 0]))
            mutado[i] += ruido
            mutado[i] = np.clip(mutado[i], limites_completos[i, 0], limites_completos[i, 1])
    return mutado

def filtrar_contraejemplos(poblacion, aptitudes, n_pares=40, dist_minima=0.5):
    """Filtrar contraejemplos válidos asegurando una separación mínima de salida.

    Ordena los individuos por aptitud y descarta aquellos excesivamente cercanos
    a soluciones ya aceptadas para diversificar la frontera.
    """
    # Descartar pares no válidos (misma clase)
    indices_validos = np.where(aptitudes < 900000)[0]
    if len(indices_validos) == 0:
        return []
    
    pob_valida = poblacion[indices_validos]
    aptitudes_validas = aptitudes[indices_validos]
    
    # Clasificar la población por mejor aptitud
    orden = np.argsort(aptitudes_validas)
    pob_ordenada = pob_valida[orden]
    
    d_dimensiones = pob_ordenada.shape[1] // 2
    
    pares_seleccionados = []
    
    # Filtrar vecinos redundantes
    for individuo in pob_ordenada:
        punto_orig = individuo[:d_dimensiones]
        cerca = False
        
        for sel in pares_seleccionados:
            punto_guardado = sel[:d_dimensiones]
            # Descartar si el punto es muy cercano a una solución previa
            if np.linalg.norm(punto_orig - punto_guardado) < dist_minima:
                cerca = True
                break
                
        if not cerca:
            pares_seleccionados.append(individuo)
            if len(pares_seleccionados) == n_pares:
                break
                
    return np.array(pares_seleccionados)

def algoritmo_genetico(modelo, limites, nombres_caracteristicas, scaler, tamano_poblacion=300, generaciones=150, tasa_mutacion=0.2, num_pares=40):
    """Ejecutar la optimización del Algoritmo Genético para localizar contraejemplos.

    Inicializa la población, ejecuta los ciclos de evaluación, selección, cruce y mutación,
    conserva la élite y retorna los contraejemplos.
    """
    d_caracteristicas = limites.shape[0]

    # Inicializar la población inicial
    poblacion = inicializar_poblacion(tamano_poblacion, limites, d_caracteristicas)

    # Calcular límites
    limites_completos = np.vstack([limites, limites])
    
    # Estimar la distancia de vecindario
    dist_minima_salida = np.mean(limites[:, 1] - limites[:, 0]) * 0.1
    
    historial_mejores = []
    elite = None
    mejor_aptitud = float('inf')
    
    print("\nIniciando algoritmo genético...\n")
    # Ciclos de optimización
    for gen in range(generaciones):
        # Evaluar la población actual
        aptitudes = evaluar_poblacion(poblacion, modelo, d_caracteristicas, nombres_caracteristicas, sigma_share=1.0, scaler=scaler)
        
        # Mejor individuo del ciclo
        indice_min_aptitud = np.argmin(aptitudes)
        mejor_aptitud_gen = aptitudes[indice_min_aptitud]
        
        # Conservar el mejor individuo
        if mejor_aptitud_gen < mejor_aptitud:
            mejor_aptitud = mejor_aptitud_gen
            elite = poblacion[indice_min_aptitud].copy()
            
        historial_mejores.append(mejor_aptitud)
        
        # Imprimir métricas de avance
        if gen % 10 == 0 or gen == generaciones - 1:
            valor = mejor_aptitud if mejor_aptitud < 900000 else "Aún no hay pares válidos"
            print(f"Generación {gen}: Mejor Distancia = {valor if isinstance(valor, str) else round(valor, 4)}")

        nueva_poblacion = []
        # Conservar la élite
        if mejor_aptitud < 900000:
            nueva_poblacion.extend([elite, elite])
            
        # Generar nuevos descendientes
        while len(nueva_poblacion) < tamano_poblacion:
            p1 = seleccion_torneo(poblacion, aptitudes)
            p2 = seleccion_torneo(poblacion, aptitudes)
            
            h1, h2 = cruce(p1, p2, alfa=np.random.rand())
            h1 = mutar(h1, limites_completos, tasa_mutacion)
            h2 = mutar(h2, limites_completos, tasa_mutacion)
            
            nueva_poblacion.extend([h1, h2])
            
        poblacion = np.array(nueva_poblacion[:tamano_poblacion])
        
    # Reevaluar la población para el filtrado
    puntos_a_scaled = poblacion[:, :d_caracteristicas]
    puntos_b_scaled = poblacion[:, d_caracteristicas:]
    df_a = pd.DataFrame(scaler.inverse_transform(puntos_a_scaled), columns=nombres_caracteristicas)
    df_b = pd.DataFrame(scaler.inverse_transform(puntos_b_scaled), columns=nombres_caracteristicas)
    clases_a = modelo.predict(df_a)
    clases_b = modelo.predict(df_b)
    
    aptitudes_finales = np.linalg.norm(puntos_a_scaled - puntos_b_scaled, axis=1)
    aptitudes_finales[clases_a == clases_b] += 999999.0
    
    # Extraer contraejemplos válidos
    pares_scaled = filtrar_contraejemplos(poblacion, aptitudes_finales, n_pares=num_pares, dist_minima=dist_minima_salida)
    
    # Desescalar los contraejemplos
    pares_raw = []
    for par in pares_scaled:
        p_a = scaler.inverse_transform([par[:d_caracteristicas]])[0]
        p_b = scaler.inverse_transform([par[d_caracteristicas:]])[0]
        pares_raw.append(np.concatenate([p_a, p_b]))
    
    return np.array(pares_raw), historial_mejores

def exportar_resultados(pares, modelo, nombres_caracteristicas, archivo_csv="datos/procesados/contraejemplos.csv"):
    """Exportar y formatear las parejas a un archivo CSV.

    Calcula la variación (delta), detecta qué variables cambiaron,
    agrega la clase predicha y exporta al fichero CSV.
    """
    d_dimension = len(nombres_caracteristicas)
    filas = []
    
    # Formatear cada par
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
    import argparse
    import os
    from src.utils.hiperparametros import obtener_hiperparametros
    
    parser = argparse.ArgumentParser(description="Algoritmo Genético para la generación de contraejemplos.")
    parser.add_argument("--train", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset de entrenamiento (default: datos/originales/diabetes.csv)")
    parser.add_argument("--test", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset de prueba (default: datos/originales/diabetes.csv)")
    parser.add_argument("--modelo-path", type=str, default="modelos/modelo.joblib", help="Ruta para guardar el modelo entrenado (default: modelos/modelo.joblib)")
    parser.add_argument("--salida", type=str, default="datos/procesados/contraejemplos.csv", help="Ruta para exportar los resultados (default: datos/procesados/contraejemplos.csv)")
    parser.add_argument("--modelo", type=str, default="mlp", choices=["svm", "mlp", "arbol_decision"], help="Tipo de clasificador base (default: mlp)")
    parser.add_argument("--tamano-poblacion", type=int, default=500, help="Tamaño de la población (default: 500)")
    parser.add_argument("--generaciones", type=int, default=300, help="Número de generaciones (default: 300)")
    parser.add_argument("--tasa-mutacion", type=float, default=0.25, help="Tasa de mutación (default: 0.25)")
    parser.add_argument("--num-pares", type=int, default=50, help="Número de contraejemplos a generar (default: 50)")
    
    args = parser.parse_args()
    
    # Entrenar clasificador
    modelo_entrenado, limites_datos, nombres_dim, scaler_genetico = entrenar_clasificador(
        args.train, args.test, args.modelo
    )
    
    # Guardar modelo entrenado
    os.makedirs(os.path.dirname(args.modelo_path), exist_ok=True)
    joblib.dump({"modelo": modelo_entrenado, "nombres": nombres_dim}, args.modelo_path)
    
    # Cargar hiperparámetros
    params_ga, _ = obtener_hiperparametros(args.train, args.modelo)

    pop_final = args.tamano_poblacion if args.tamano_poblacion != 500 else params_ga["tamano_poblacion"]
    gen_final = args.generaciones if args.generaciones != 300 else params_ga["generaciones"]
    mut_final = args.tasa_mutacion if args.tasa_mutacion != 0.25 else params_ga["tasa_mutacion"]
    
    # Algoritmo genético
    pares_finales, historial = algoritmo_genetico(
        modelo=modelo_entrenado, 
        limites=limites_datos,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler_genetico,
        tamano_poblacion=pop_final, 
        generaciones=gen_final,
        tasa_mutacion=mut_final,
        num_pares=args.num_pares
    )
    
    # Guardar resultados finales
    if len(pares_finales) > 0:
        os.makedirs(os.path.dirname(args.salida), exist_ok=True)
        exportar_resultados(pares_finales, modelo_entrenado, nombres_dim, args.salida)
    else:
        print("\nNo se encontró ningún par válido.")
