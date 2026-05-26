"""Obtención de hiperparámetros óptimos.

Define y carga las configuraciones recomendadas para algoritmos Genéticos y
Growing Spheres según el clasificador y conjunto de datos de interés.
"""

def obtener_hiperparametros(dataset_name_or_path, tipo_modelo):
    """Obtener hiperparámetros recomendados para el dataset y clasificador dados.

    Normaliza las cadenas de entrada, evalúa las condiciones del dataset
    y retorna los diccionarios de configuración para la búsqueda global y local.
    """
    ds = str(dataset_name_or_path).lower()
    mod = str(tipo_modelo).lower()

    # Definir hiperparámetros por defecto para el Algoritmo Genético
    params_ga = {
        "tamano_poblacion": 200,
        "generaciones": 40,
        "tasa_mutacion": 0.30
    }
    
    # Asignar parámetros por defecto para Growing Spheres
    if "svm" in mod:
        params_gs = {
            "muestras": 100,
            "ancho_banda": 0.2, 
            "max_iters": 20
        }
    elif "mlp" in mod:
        params_gs = {
            "muestras": 300,
            "ancho_banda": 0.5, 
            "max_iters": 20
        }
    else: # arbol_decision
        params_gs = {
            "muestras": 500,
            "ancho_banda": 0.8, 
            "max_iters": 20
        }

    # Sobreescribir configuraciones específicas para el dataset Diabetes
    if "diabetes" in ds:
        if "svm" in mod:
            params_ga = {
                "tamano_poblacion": 200,
                "generaciones": 40,
                "tasa_mutacion": 0.15
            }
            params_gs = {
                "muestras": 100,
                "ancho_banda": 0.8, 
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
                "ancho_banda": 0.8,
                "max_iters": 20
            }
        else: # arbol_decision
            params_ga = {
                "tamano_poblacion": 200,
                "generaciones": 80,
                "tasa_mutacion": 0.15
            }
            params_gs = {
                "muestras": 500,
                "ancho_banda": 0.5,
                "max_iters": 60
            }
            
    return params_ga, params_gs
