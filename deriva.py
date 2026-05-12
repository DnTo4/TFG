import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    y_raw = df[df.columns[-1]]
    if y_raw.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw.astype(int)
    X = df.drop(columns=[df.columns[-1]])
    X = pd.get_dummies(X)
    return X, y

def entrenar_modelo(X, y, tipo_modelo="mlp"):
    scaler = StandardScaler()
    if tipo_modelo == "svm":
        clf = SVC(probability=True, random_state=42)
    else: # mlp
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
    
    modelo = make_pipeline(scaler, clf)
    modelo.fit(X, y)
    return modelo

def frontera_biseccion(modelo, X, y, num_puntos=50):
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
        if c0 == c1: continue
            
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
    probas = modelo.predict_proba(X)
    return np.max(probas, axis=1)

def cuantificar_deriva_espacial(anclajes_b0, modelo_nuevo, cols):
    d_dims = len(cols)
    distancias = []
    paso_radio = 0.05
    radio_max  = 5.0 
    nPts_esfera = 300
    
    for b in anclajes_b0:
        clase_original = modelo_nuevo.predict(pd.DataFrame([b], columns=cols))[0]
        encontrado = False
        r = 0.01
        while r <= radio_max:
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
            distancias.append(radio_max) 
    return np.mean(distancias)

def main(dataset_path, tipo_modelo):
    print("=== Fase 4: Análisis de Concept Drift (Deriva) ===")
    print(f"[*] Dataset : {dataset_path}")
    print(f"[*] Modelo  : {tipo_modelo.upper()}")
    
    try:
        X, y = cargar_datos(dataset_path)
        print(f"[*] Datos procesados -> [{X.shape[0]} instancias, {X.shape[1]} variables.]")
    except Exception as e:
        print(f"[-] Error cargando los datos: {e}")
        return

    print("[*] Entrenando modelo base...")
    modelo_base = entrenar_modelo(X, y, tipo_modelo)
    
    print("[*] Buscando puntos tangenciales en la frontera (Bisección)...")
    B_0 = frontera_biseccion(modelo_base, X, y, num_puntos=30)
    
    if len(B_0) == 0:
        print("[-] Fatal: No se lograron converger puntos tangenciales entre clases.")
        return

    print(f"[+] Encontrados {len(B_0)} anclajes de frontera.")
    lejanias_orig = lejania_datos(modelo_base, X)
    porcentajes_corte = [0, 10, 20, 30, 40, 50, 60]
    registro_deriva = []
    
    for P in porcentajes_corte:
        if P == 0:
            deriva = 0.0
            registro_deriva.append(deriva)
            print(f" -> Puntos eliminados: {P:2d}% | Desplazamiento de frontera: {deriva:.4f}")
            continue
            
        umbral_corte = np.percentile(lejanias_orig, 100 - P) 
        mascara_supervivientes = lejanias_orig <= umbral_corte
        
        X_vivo = X[mascara_supervivientes]
        y_vivo = y[mascara_supervivientes]
        
        modelo_cortado = entrenar_modelo(X_vivo, y_vivo, tipo_modelo)
        deriva_media = cuantificar_deriva_espacial(B_0, modelo_cortado, X.columns)
        registro_deriva.append(deriva_media)
        
        print(f" -> Puntos eliminados: {P:2d}% | Desplazamiento de frontera: {deriva_media:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(porcentajes_corte, registro_deriva, marker='o', color='#4CAF50', linewidth=2.5)
    plt.fill_between(porcentajes_corte, registro_deriva, color='#4CAF50', alpha=0.15)
    plt.grid(color='#E0E0E0', linestyle='--')
    plt.title(f"Análisis de Deriva Espacial - {tipo_modelo.upper()}", fontweight='bold')
    plt.xlabel("Porcentaje de Puntos Lejanos Eliminados (%)")
    plt.ylabel("Distancia de Desplazamiento Media")
    plt.xlim(-2, 65)
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar la Fase 4: Análisis de Concept Drift.")
    parser.add_argument("--dataset", type=str, default="datos/originales/iris.data", help="Ruta al dataset")
    parser.add_argument("--modelo", type=str, default="mlp", choices=["svm", "mlp"], help="Modelo a utilizar (perceptron omitido porque requiere probas)")
    args = parser.parse_args()
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main(args.dataset, args.modelo)
