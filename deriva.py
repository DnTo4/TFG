import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUTA_DATASET = "iris.data"

def cargar_datos(ruta):
    """
    Carga un conjunto de datos desde un CSV, codifica etiquetas y maneja variables categóricas.

    Args:
        ruta (str): Ruta del archivo de datos.

    Returns:
        tuple: (X, y) donde "X" son las características (DataFrame) e "y" las etiquetas (array).
    """
    df = pd.read_csv(ruta)
    y_raw = df[df.columns[-1]]
    
    # Codificación de etiquetas si son texto
    if y_raw.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw.astype(int)
        
    X = df.drop(columns=[df.columns[-1]])
    X = pd.get_dummies(X) # One-hot encoding para variables categóricas
    return X, y

def entrenar_modelo(X, y):
    """
    Crea un pipeline que escala los datos y entrena un modelo.

    Args:
        X (DataFrame): Datos de entrenamiento.
        y (array): Etiquetas de clase.

    Returns:
        sklearn.pipeline.Pipeline: Modelo entrenado y escalado.
    """
    scaler = StandardScaler()
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
    modelo = make_pipeline(scaler, clf)
    modelo.fit(X, y)
    return modelo

def frontera_biseccion(modelo, X, y, num_puntos=50):
    """
    Encuentra puntos exactos sobre la frontera de decisión usando el método de bisección.
    
    Toma dos puntos aleatorios de clases distintas y busca el punto medio exacto donde 
    la predicción cambia.

    Args:
        modelo: Clasificador entrenado.
        X (DataFrame): Datos originales para muestreo.
        y (array): Etiquetas originales.
        num_puntos (int): Cantidad de puntos de anclaje a encontrar.

    Returns:
        np.ndarray: Coordenadas de los puntos localizados en la frontera.
    """
    cols = X.columns
    anclajes = []
    np.random.seed(42)
    
    intentos = 0
    X_vals = X.values if hasattr(X, 'values') else X
    
    while len(anclajes) < num_puntos and intentos < num_puntos * 20:
        intentos += 1
        p0 = X_vals[np.random.randint(0, len(X_vals))]
        p1 = X_vals[np.random.randint(0, len(X_vals))]
        
        c0 = modelo.predict(pd.DataFrame([p0], columns=cols))[0]
        c1 = modelo.predict(pd.DataFrame([p1], columns=cols))[0]
        
        # Solo buscamos la frontera entre clases diferentes
        if c0 == c1:
            continue
            
        # Refinamiento mediante bisección (15 iteraciones para alta precisión)
        for _ in range(15):
            pm = (p0 + p1) / 2.0
            cm = modelo.predict(pd.DataFrame([pm], columns=cols))[0]
            
            if cm == c0:
                p0 = pm
            else:
                p1 = pm
                
        anclajes.append((p0 + p1) / 2.0)
        
    return np.array(anclajes)

def lejania_datos(modelo, X):
    """
    Calcula el nivel de 'confianza' del modelo para cada punto.
    
    Un valor alto indica que el punto está lejos de la frontera,
    mientras que un valor cercano a 1/n_clases indica cercanía a la frontera.

    Returns:
        np.ndarray: Probabilidad máxima asignada a cada instancia.
    """
    probas = modelo.predict_proba(X)
    return np.max(probas, axis=1)

def cuantificar_deriva_espacial(anclajes_b0, modelo_nuevo, cols):
    """
    Mide cuánto se ha desplazado la frontera del 'modelo_nuevo' respecto a los 
    puntos originales mediante una expansión radial.

    Args:
        anclajes_b0 (np.ndarray): Puntos que estaban en la frontera original.
        modelo_nuevo: El modelo reentrenado.
        cols: Nombres de las columnas de X.

    Returns:
        float: Distancia media de desplazamiento.
    """
    d_dims = len(cols)
    distancias = []
    paso_radio = 0.05
    radio_max  = 5.0 
    nPts_esfera = 300 # Densidad de muestreo en la superficie de la esfera
    
    for b in anclajes_b0:
        clase_original = modelo_nuevo.predict(pd.DataFrame([b], columns=cols))[0]
        encontrado = False
        r = 0.01
        
        # Expandir un radio de búsqueda desde el punto de anclaje
        while r <= radio_max:
            # Generar vectores unitarios aleatorios en d-dimensiones
            v = np.random.normal(size=(nPts_esfera, d_dims))
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            pts = b + (v * r)
            
            clases_barrido = modelo_nuevo.predict(pd.DataFrame(pts, columns=cols))
            
            # Si algún punto de la esfera cambia de clase, es de la nueva frontera
            if np.any(clases_barrido != clase_original):
                distancias.append(r)
                encontrado = True
                break
            r += paso_radio
            
        if not encontrado:
            distancias.append(radio_max) 
            
    return np.mean(distancias)

def main():
    """
    Flujo principal: carga datos, entrena un modelo base, elimina puntos
    periféricos de forma iterativa y mide la deriva de la frontera.
    """
    try:
        X, y = cargar_datos(RUTA_DATASET)
        print(f"\nDataset '{RUTA_DATASET}' procesado -> [{X.shape[0]} instancias, {X.shape[1]} variables.]")
    except Exception as e:
        print(f"Error cargando los datos: {e}")
        return

    # Establecer el estado inicial
    modelo_base = entrenar_modelo(X, y)
    B_0 = frontera_biseccion(modelo_base, X, y, num_puntos=50)
    
    if len(B_0) == 0:
        print("Fatal: No se lograron converger puntos tangenciales entre clases.")
        return

    lejanias_orig = lejania_datos(modelo_base, X)
    porcentajes_corte = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    registro_deriva = []
    
    # Eliminar puntos lejanos y observar el cambio
    for P in porcentajes_corte:
        if P == 0:
            deriva = 0.0
            registro_deriva.append(deriva)
            print(f" -> Puntos eliminados: {P:2d}% | Desplazamiento de frontera: {deriva:.4f}")
            continue
            
        # Identificar el umbral para eliminar el P% de datos
        umbral_corte = np.percentile(lejanias_orig, 100 - P) 
        mascara_supervivientes = lejanias_orig <= umbral_corte
        
        X_vivo = X[mascara_supervivientes]
        y_vivo = y[mascara_supervivientes]
        
        # Reentrenar con el dataset reducido
        modelo_cortado = entrenar_modelo(X_vivo, y_vivo)
        
        # Medir cuánto se movió el borde respecto a B_0
        deriva_media = cuantificar_deriva_espacial(B_0, modelo_cortado, X.columns)
        registro_deriva.append(deriva_media)
        
        print(f" -> Puntos eliminados: {P:2d}% | Desplazamiento de frontera: {deriva_media:.4f}")

    # Visualización de resultados
    plt.figure(figsize=(8, 5))
    plt.plot(porcentajes_corte, registro_deriva, marker='o', color='#4CAF50', linewidth=2.5)
    plt.fill_between(porcentajes_corte, registro_deriva, color='#4CAF50', alpha=0.15)
    plt.grid(color='#E0E0E0', linestyle='--')
    
    plt.title("Análisis de Deriva Espacial de la Frontera", fontweight='bold')
    plt.xlabel("Porcentaje de Puntos Lejanos Eliminados (%)")
    plt.ylabel("Distancia de Desplazamiento Media")
    
    plt.xlim(-2, 85)
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
