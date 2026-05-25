"""
Módulo de utilidades para la resolución de hiperparámetros de GA y GS.
"""

def obtener_hiperparametros(dataset_name_or_path, tipo_modelo):
    """
    Retorna los hiperparámetros óptimos para GA y GS según el dataset y clasificador.
    
    Args:
        dataset_name_or_path (str): Nombre o ruta del dataset.
        tipo_modelo (str): Tipo de clasificador ('svm', 'mlp', 'arbol_decision').
        
    Returns:
        tuple: (params_ga, params_gs)
            params_ga: dict con {'tamano_poblacion', 'generaciones', 'tasa_mutacion'}
            params_gs: dict con {'muestras', 'ancho_banda', 'max_iters'}
    """
    ds = str(dataset_name_or_path).lower()
    mod = str(tipo_modelo).lower()
    
    # Por defecto y para conjuntos sintéticos (como moons), se usan los parámetros de Iris.
    # Iris GA: 200 individuos, 40 generaciones, 30% mutación (para todos SVM, MLP, Árbol de Decisión)
    params_ga = {
        "tamano_poblacion": 200,
        "generaciones": 40,
        "tasa_mutacion": 0.30
    }
    
    # Iris GS por defecto:
    if "svm" in mod:
        params_gs = {
            "muestras": 100,
            "ancho_banda": 0.2, # 20%
            "max_iters": 20
        }
    elif "mlp" in mod:
        params_gs = {
            "muestras": 300,
            "ancho_banda": 0.5, # 50%
            "max_iters": 20
        }
    else: # arbol_decision / arbol
        params_gs = {
            "muestras": 500,
            "ancho_banda": 0.8, # 80%
            "max_iters": 20
        }
        
    # Si es Diabetes:
    if "diabetes" in ds:
        # GA para Diabetes:
        if "svm" in mod:
            params_ga = {
                "tamano_poblacion": 200,
                "generaciones": 40,
                "tasa_mutacion": 0.15
            }
            params_gs = {
                "muestras": 100,
                "ancho_banda": 0.8, # 80%
                "max_iters": 60
            }
        elif "mlp" in mod:
            params_ga = {
                "tamano_poblacion": 200,
                "generaciones": 120,
                "tasa_mutacion": 0.15
            }
            params_gs = {
                "muestras": 500,
                "ancho_banda": 0.8, # 80%
                "max_iters": 20
            }
        else: # arbol_decision / arbol
            params_ga = {
                "tamano_poblacion": 200,
                "generaciones": 80,
                "tasa_mutacion": 0.15
            }
            params_gs = {
                "muestras": 500,
                "ancho_banda": 0.5, # 50%
                "max_iters": 60
            }
            
    return params_ga, params_gs
