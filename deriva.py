import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

"""Análisis y cuantificación de la deriva espacial en fronteras de decisión.

Permite evaluar cómo se desplazan las fronteras de clasificadores al reducir 
progresivamente el conjunto de datos de entrenamiento mediante poda de instancias.
"""

def cargar_datos(ruta):
    """Cargar y preprocesar el conjunto de datos para el análisis.

    Lee el archivo CSV, detecta la codificación de etiquetas y normaliza
    las variables categóricas si es necesario.
    """
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
    """Entrenar un clasificador.

    Configura y ajusta clasificadores de tipo SVM, Árbol de Decisión o MLP
    utilizando escalado de características.
    """
    scaler = StandardScaler()
    if tipo_modelo == "svm":
        clf = SVC(probability=True, random_state=42)
    elif tipo_modelo == "arbol_decision":
        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
    else: # mlp
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
    
    modelo = make_pipeline(scaler, clf)
    modelo.fit(X, y)
    return modelo

def frontera_biseccion(modelo, X, y, num_puntos=50):
    """Buscar puntos localizados sobre la frontera de decisión.

    Utiliza el método numérico de bisección entre pares de instancias del conjunto
    que pertenecen a diferentes clases predichas.
    """
    cols = X.columns
    anclajes = []
    np.random.seed(42)
    intentos = 0
    X_vals = X.values if hasattr(X, 'values') else X
    
    # Iterar buscando pares de puntos con predicciones distintas
    while len(anclajes) < num_puntos and intentos < num_puntos * 20:
        intentos += 1
        p0 = X_vals[np.random.randint(0, len(X_vals))]
        p1 = X_vals[np.random.randint(0, len(X_vals))]
        c0 = modelo.predict(pd.DataFrame([p0], columns=cols))[0]
        c1 = modelo.predict(pd.DataFrame([p1], columns=cols))[0]
        if c0 == c1: continue
            
        # Refinar el punto medio mediante aproximación por bisección
        for _ in range(15):
            pm = (p0 + p1) / 2.0
            cm = modelo.predict(pd.DataFrame([pm], columns=cols))[0]
            if cm == c0:
                p0 = pm
            else:
                p1 = pm
        anclajes.append((p0 + p1) / 2.0)
    return np.array(anclajes)

def lejanias_orig_datos(modelo, X):
    """Calcular la lejanía o certeza de predicción para cada instancia.

    Obtiene la probabilidad máxima asignada por el clasificador para cada
    punto del conjunto.
    """
    try:
        probas = modelo.predict_proba(X)
        return np.max(probas, axis=1)
    except AttributeError:
        # Fallback a decision_function si predict_proba no está disponible (ej. SVM preentrenado con probability=False)
        if hasattr(modelo, "decision_function"):
            dec = modelo.decision_function(X)
            if len(dec.shape) > 1 and dec.shape[1] > 1:
                return np.max(np.abs(dec), axis=1)
            else:
                return np.abs(dec)
        else:
            return np.ones(len(X))


def cuantificar_deriva_espacial(anclajes_b0, modelo_nuevo, cols):
    """Medir el desplazamiento espacial de la frontera original en el nuevo modelo.

    Realiza un barrido radial mediante muestreo esférico alrededor de cada punto de anclaje
    para encontrar la distancia mínima hasta la nueva frontera de decisión.
    """
    d_dims = len(cols)
    distancias = []
    paso_radio = 0.05
    radio_max  = 2.0 
    nPts_esfera = 300
    
    # Estimar el radio de deriva para cada punto de anclaje
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
    """Ejecutar análisis de deriva espacial.

    Entrena los clasificadores, localiza sus fronteras iniciales, aplica podas sucesivas
    en función de la certidumbre de los datos y cuantifica la deriva espacial resultante.
    """
    print("=== Análisis de Deriva ===")
    print(f"Dataset: {dataset_path}")
    print(f"Modelo Complejo: {tipo_modelo.upper()}")
    print("Modelo Simplificado: ÁRBOL DE DECISIÓN")
    
    # Cargar el conjunto de datos
    try:
        X, y = cargar_datos(dataset_path)
        print(f"Datos procesados -> [{X.shape[0]} instancias, {X.shape[1]} variables.]")
    except Exception as e:
        print(f"Error cargando los datos: {e}")
        return

    print("\nCargando o entrenando modelos base...")
    import joblib
    
    # Intentar cargar el modelo complejo preentrenado
    modelo_complex = None
    if os.path.exists("modelos/modelo.joblib"):
        try:
            bundle = joblib.load("modelos/modelo.joblib")
            if set(bundle.get("nombres", [])) == set(X.columns):
                modelo_complex = bundle["modelo"]
                print(f"Cargado modelo complejo preentrenado desde 'modelos/modelo.joblib'.")
        except Exception as e:
            pass
            
    if modelo_complex is None:
        print(f"Entrenando modelo complejo ({tipo_modelo.upper()})...")
        modelo_complex = entrenar_modelo(X, y, tipo_modelo)
        
    # Intentar cargar el modelo simplificado preentrenado
    modelo_simple = None
    if os.path.exists("modelos/modelo_simplificado.joblib"):
        try:
            bundle_simple = joblib.load("modelos/modelo_simplificado.joblib")
            if set(bundle_simple.get("nombres", [])) == set(X.columns):
                modelo_simple = bundle_simple["modelo"]
                print("Cargado modelo simplificado preentrenado desde 'modelos/modelo_simplificado.joblib'.")
        except Exception as e:
            pass
            
    if modelo_simple is None:
        print("Entrenando modelo simple (Árbol de Decisión)...")
        modelo_simple = entrenar_modelo(X, y, "arbol_decision")

    
    print("\nBuscando puntos tangenciales...")
    B_0_complex = frontera_biseccion(modelo_complex, X, y, num_puntos=30)
    B_0_simple = frontera_biseccion(modelo_simple, X, y, num_puntos=30)
    
    if len(B_0_complex) == 0 or len(B_0_simple) == 0:
        print("Error: No se lograron converger puntos tangenciales en alguna de las fronteras.")
        return

    print(f"Encontrados {len(B_0_complex)} anclajes para la Caja Negra ({tipo_modelo.upper()}).")
    print(f"Encontrados {len(B_0_simple)} anclajes para el Árbol de Decisión.")

    # Calcular certezas en los puntos originales
    lejanias_orig = lejanias_orig_datos(modelo_complex, X)
    porcentajes_corte = [0, 10, 20, 30, 40, 50, 60]
    
    registro_deriva_complex = []
    registro_deriva_simple = []
    
    print("\nEjecutando poda y reentrenamiento...")
    # Evaluar la deriva para cada porcentaje de poda configurado
    for P in porcentajes_corte:
        if P == 0:
            drift_complex = 0.0
            drift_simple = 0.0
            registro_deriva_complex.append(drift_complex)
            registro_deriva_simple.append(drift_simple)
            print(f"Puntos eliminados: {P:2d}% | Deriva Caja Negra: {drift_complex:.4f} | Deriva Árbol: {drift_simple:.4f}")
            continue
            
        umbral_corte = np.percentile(lejanias_orig, 100 - P) 
        mascara_supervivientes = lejanias_orig <= umbral_corte
        
        X_vivo = X[mascara_supervivientes]
        y_vivo = y[mascara_supervivientes]
        
        # Reentrenar modelos podados
        modelo_complex_cortado = entrenar_modelo(X_vivo, y_vivo, tipo_modelo)
        modelo_simple_cortado = entrenar_modelo(X_vivo, y_vivo, "arbol_decision")
        
        # Calcular derivas espaciales con respecto a los anclajes de partida
        drift_complex = cuantificar_deriva_espacial(B_0_complex, modelo_complex_cortado, X.columns)
        drift_simple = cuantificar_deriva_espacial(B_0_simple, modelo_simple_cortado, X.columns)
        
        registro_deriva_complex.append(drift_complex)
        registro_deriva_simple.append(drift_simple)
        
        print(f"Puntos eliminados: {P:2d}% | Deriva Caja Negra: {drift_complex:.4f} | Deriva Árbol: {drift_simple:.4f}")

    # Imprimir informe
    print("\n=========================================================")
    print("                RESULTADOS DE DERIVA")
    print("=========================================================")
    reporte_data = {
        "Porcentaje de puntos eliminados": [f"{p}%" for p in porcentajes_corte],
        f"Deriva Caja Negra ({tipo_modelo.upper()})": [f"{d:.4f}" for d in registro_deriva_complex],
        "Deriva Árbol de Decisión": [f"{d:.4f}" for d in registro_deriva_simple]
    }
    df_reporte = pd.DataFrame(reporte_data)
    print(df_reporte.to_string(index=False))
    print("=========================================================")

    # Dibujar gráfica de comparación temporal y espacial de deriva
    print("\nGenerando gráfico comparativo...")
    plt.figure(figsize=(10, 6))
    
    # Graficar curva de la caja negra
    plt.plot(porcentajes_corte, registro_deriva_complex, marker='o', linestyle='-', linewidth=2.5, color='#9C27B0', label=f'Caja Negra ({tipo_modelo.upper()})')
    plt.fill_between(porcentajes_corte, registro_deriva_complex, color='#9C27B0', alpha=0.10)
    
    # Graficar curva del modelo subrogado
    plt.plot(porcentajes_corte, registro_deriva_simple, marker='s', linestyle='--', linewidth=2.5, color='#008080', label='Árbol de Decisión')
    plt.fill_between(porcentajes_corte, registro_deriva_simple, color='#008080', alpha=0.10)
    
    plt.grid(color='#E0E0E0', linestyle='--', alpha=0.7)
    plt.title(f"Análisis de Deriva Espacial (Dataset: {os.path.basename(dataset_path)})", fontweight='bold', fontsize=12, pad=15)
    plt.xlabel("Porcentaje de puntos eliminados", fontweight='bold')
    plt.ylabel("Desplazamiento medio de la frontera", fontweight='bold')
    plt.xlim(-2, 65)
    plt.ylim(bottom=0.0)
    plt.legend(frameon=True, edgecolor='#E0E0E0', loc='upper left')
    plt.tight_layout()
    
    # Exportar imagen
    os.makedirs("resultados", exist_ok=True)
    grafico_path = "resultados/comparativa_deriva.png"
    plt.savefig(grafico_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en '{grafico_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis de Deriva.")
    parser.add_argument("--dataset", type=str, default="datos/originales/iris.data", help="Ruta al dataset")
    parser.add_argument("--modelo", type=str, default="mlp", choices=["svm", "mlp"], help="Modelo a utilizar (arbol_decision omitido porque requiere probas)")
    args = parser.parse_args()
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main(args.dataset, args.modelo)
