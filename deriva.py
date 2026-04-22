import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUTA_DATASET = "iris.data"

def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    y_raw = df[df.columns[-1]]
    
    if y_raw.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw.astype(int)
        
    X = df.drop(columns=[df.columns[-1]])
    X = pd.get_dummies(X) # Por si hay características categóricas
    return X, y

def entrenar_modelo(X, y):
    scaler = StandardScaler()
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
    modelo = make_pipeline(scaler, clf)
    modelo.fit(X, y)
    return modelo

def anclar_frontera_biseccion(modelo, X, y, num_puntos=50):
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
        
        # Ignorar si ambas muestran la misma clase
        if c0 == c1:
            continue
            
        # Bisección de alta precisión geométrica (15 pasos)
        for _ in range(15):
            pm = (p0 + p1) / 2.0
            cm = modelo.predict(pd.DataFrame([pm], columns=cols))[0]
            
            if cm == c0:
                p0 = pm
            else:
                p1 = pm
                
        anclajes.append((p0 + p1) / 2.0)
        
    return np.array(anclajes)

def calcular_lejania_datos(modelo, X):
    probas = modelo.predict_proba(X)
    return np.max(probas, axis=1)

def cuantificar_deriva_espacial(anclajes_b0, modelo_nuevo, cols):
    d_dims = len(cols)
    distancias = []
    
    # Tolerancia/Salto geodésico
    paso_radio = 0.05
    radio_max  = 5.0 
    nPts_esfera = 300
    
    for b in anclajes_b0:
        clase_original = modelo_nuevo.predict(pd.DataFrame([b], columns=cols))[0]
        encontrado = False
        r = 0.01
        
        # Expansión radial isotrópica
        while r <= radio_max:
            # Vectors aleatorios mapeados a norm=1 unitaria multivariada
            v = np.random.normal(size=(nPts_esfera, d_dims))
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            pts = b + (v * r)
            
            clases_barrido = modelo_nuevo.predict(pd.DataFrame(pts, columns=cols))
            
            if np.any(clases_barrido != clase_original):
                distancias.append(r)
                encontrado = True
                break
            r += paso_radio
            
        if not encontrado:
            distancias.append(radio_max) # Si la frontera fue engullida dramáticamente
            
    return np.mean(distancias)

def main():

    try:
        X, y = cargar_datos(RUTA_DATASET)
        print(f"\nDataset '{RUTA_DATASET}' procesado -> [{X.shape[0]} instancias, {X.shape[1]} variables.]")
    except Exception as e:
        print(f"Error cargando los datos: {e}")
        return

    modelo_base = entrenar_modelo(X, y)
    
    B_0 = anclar_frontera_biseccion(modelo_base, X, y, num_puntos=50)
    if len(B_0) == 0:
        print("Fatal: No se lograron converger puntos tangenciales entre clases.")
        return

    lejanias_orig = calcular_lejania_datos(modelo_base, X)
    
    porcentajes_corte = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    registro_deriva = []
    
    for P in porcentajes_corte:
        if P == 0: # Control baseline (si nos comparamos contra el mismo modelo es casi 0 teórico)
            deriva = 0.0
            registro_deriva.append(deriva)
            print(f" -> Puntos eliminados: {P:2d}% | Desplazamiento de frontera: {deriva:.4f}")
            continue
            
        # Nos disponemos a eliminar el P% de datos más lejanos/seguros
        umbral_corte = np.percentile(lejanias_orig, 100 - P) 
        
        # Nos quedamos con los datos cercanos a la frontera original
        mascara_supervivientes = lejanias_orig <= umbral_corte
        
        X_vivo = X[mascara_supervivientes]
        y_vivo = y[mascara_supervivientes]
        
        # Reentrenamos el modelo con menos datos
        modelo_ablado = entrenar_modelo(X_vivo, y_vivo)
        
        # Exploración de cuánto huyó el hiperplano localmente
        deriva_media = cuantificar_deriva_espacial(B_0, modelo_ablado, X.columns)
        registro_deriva.append(deriva_media)
        
        print(f" -> Puntos eliminados: {P:2d}% | Desplazamiento de frontera: {deriva_media:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(porcentajes_corte, registro_deriva, marker='o', markersize=7, linewidth=2.5, color='#4CAF50')
    
    plt.fill_between(porcentajes_corte, registro_deriva, color='#4CAF50', alpha=0.15)
    plt.grid(color='#E0E0E0', linestyle='--', linewidth=1)
    
    plt.title("Desplazamiento de la Frontera de Decisión", pad=15, fontweight='bold', color='#333333')
    plt.xlabel("Porcentaje de Puntos Eliminados (%)", labelpad=10, weight='semibold')
    plt.ylabel("Distancia de Desplazamiento Media", labelpad=10, weight='semibold')
    
    plt.xlim(-2, 85)
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig("deriva_analisis.png", dpi=250)
    print("Archivo guardado como 'deriva_analisis.png'.")
    plt.show()

if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
