import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Configuración global
CSV_PATH   = "contraejemplos_memeticos.csv"
MODEL_PATH = "modelo.joblib" 
FRONTERA   = True    # Visualizar frontera de decisión (requiere modelo)
FLECHAS    = False   # Visualizar flechas origen -> contraejemplo
USAR_SHAP  = False   # Priorizar importancia global (SHAP) vs empírica (movimiento)

def analizar_csv(path):
    """
    Lee un archivo CSV con contraejemplos y realiza un análisis estadístico completo.
    
    El análisis incluye:
    1. Estadísticas de distancia euclídea (L2).
    2. Conteo de variables modificadas para lograr el cambio de clase.
    3. Ranking de variables por frecuencia de cambio y magnitud del desplazamiento.
    4. Cálculo de un score de importancia (frecuencia * magnitud).

    Args:
        path (str): Ruta del archivo CSV generado por el algoritmo de contraejemplos.
    """
    df = pd.read_csv(path)

    print("\n==============================")
    print("ANÁLISIS DE CONTRAEJEMPLOS")
    print("==============================\n")

    n = len(df)
    print(f"Numero de contraejemplos: {n}")

    # Análisis de Distancias L2
    if "dist_l2" in df.columns:
        print("\nDistancia al contraejemplo")
        print("---------------------------")
        print(f"  media  : {df['dist_l2'].mean():.4f}")
        print(f"  minima : {df['dist_l2'].min():.4f}")
        print(f"  maxima : {df['dist_l2'].max():.4f}")
        print(f"  mediana: {df['dist_l2'].median():.4f}")

    # Análisis de sparsity
    if "num_features_changed" in df.columns:
        print("\nNumero de variables modificadas")
        print("--------------------------------")
        print(f"  media  : {df['num_features_changed'].mean():.2f}")
        print(f"  mediana: {df['num_features_changed'].median():.0f}")

        conteo = df["num_features_changed"].value_counts().sort_index()
        print("\n  Distribucion:")
        for k, v in conteo.items():
            print(f"  {k} variables -> {v} veces ({100*v/n:.1f}%)")

    # Análisis de frecuencia (cuántas veces se tocó cada variable)
    changed_cols = [c for c in df.columns if c.startswith("changed_")]
    if changed_cols:
        print("\nFrecuencia de cambio por variable")
        print("----------------------------------")
        freq = df[changed_cols].mean()
        freq.index = [c.replace("changed_", "") for c in freq.index]
        ranking_freq = freq.sort_values(ascending=False)
        for var, val in ranking_freq.items():
            print(f"  {var}: {val:.3f}")

    # Análisis de magnitud (cuánto se movió cada variable en promedio)
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    if delta_cols:
        print("\nMagnitud media del cambio")
        print("--------------------------")
        magnitud = df[delta_cols].abs().mean()
        magnitud.index = [c.replace("delta_", "") for c in magnitud.index]
        ranking_mag = magnitud.sort_values(ascending=False)
        for var, val in ranking_mag.items():
            print(f"  {var}: {val:.4f}")

    # Cálculo del Ranking Global
    nombres, score = None, None
    if changed_cols and delta_cols:
        print("\nRanking global de importancia (frecuencia x magnitud)")
        print("------------------------------------------------------")
        nombres = [c.replace("changed_", "") for c in changed_cols]
        freq    = df[changed_cols].mean().values
        mag     = df[delta_cols].abs().mean().values
        score   = freq * mag
        ranking = np.argsort(score)[::-1]

        for pos, i in enumerate(ranking, 1):
            print(
                f"  {pos:2}. {nombres[i]:<30} "
                f"frecuencia={freq[i]:.3f}  "
                f"cambio_medio={mag[i]:.4f}  "
                f"score={score[i]:.4f}"
            )

    plot_contraejemplos(df, nombres, score)

def plot_contraejemplos(df, nombres=None, score=None):
    """
    Genera una visualización 2D de los puntos originales vs. contraejemplos.
    
    Selecciona las dos variables más importantes (vía SHAP o vía Score empírico) 
    para los ejes X e Y. Si se dispone del modelo, dibuja las regiones de 
    decisión en el fondo.

    Args:
        df (pandas.DataFrame): Datos de los contraejemplos.
        nombres (list, opcional): Lista de nombres de las variables analizadas.
        score (numpy.ndarray, opcional): Scores de importancia calculados en analizar_csv.
    """
    print("\nPreparando gráfico...")
    modelo = None
    nombres_modelo = None

    # Carga del modelo y metadatos
    try:
        bundle         = joblib.load(MODEL_PATH)
        modelo         = bundle["modelo"]
        nombres_modelo = bundle["nombres"]
    except FileNotFoundError:
        print(f"Aviso: no se encontró '{MODEL_PATH}', se omite la frontera y SHAP.")

    var_a, var_b = None, None

    # Lógica de selección de variables para los ejes
    if USAR_SHAP and modelo is not None and nombres_modelo is not None:
        try:
            import shap
            X_orig = df[nombres_modelo]
            predict_fn = modelo.predict_proba if hasattr(modelo, "predict_proba") else modelo.predict
            n_samples = min(200, len(X_orig))
            X_sample = shap.sample(X_orig, n_samples)
            
            explainer = shap.Explainer(predict_fn, X_sample)
            shap_values = explainer(X_sample)
            
            vals = shap_values.values
            shap_importance = np.abs(vals).mean(axis=(0, 2)) if len(vals.shape) == 3 else np.abs(vals).mean(axis=0)

            ranking_shap = np.argsort(shap_importance)[::-1]
            var_a, var_b = nombres_modelo[ranking_shap[0]], nombres_modelo[ranking_shap[1]]
            print(f"Variables SHAP: '{var_a}' y '{var_b}'")
        except Exception as e:
            print(f"Error SHAP: {e}")

    if var_a is None or var_b is None:
        if nombres is not None and score is not None:
            print("Usando frecuencia x magnitud para los ejes.")
            ranking = np.argsort(score)[::-1]
            var_a, var_b = nombres[ranking[0]], nombres[ranking[1]]
        else:
            print("No se pudo seleccionar variables para el gráfico.")
            return

    # Preparación de datos para el plot
    orig_a, orig_b = df[var_a].values, df[var_b].values
    ce_a, ce_b = df[f"ce_{var_a}"].values, df[f"ce_{var_b}"].values
    clases_all = sorted(df["pred_orig"].unique())
    colores_base = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4"]
    colores_osc  = ["#0D47A1", "#1B5E20", "#E65100", "#4A148C", "#880E4F", "#006064"]

    plt.figure(figsize=(7, 6))

    # Renderizado de la Frontera de Decisión
    if FRONTERA and modelo is not None:
        from matplotlib.colors import ListedColormap
        pad = 0.8
        x_min, x_max = min(orig_a.min(), ce_a.min()) - pad, max(orig_a.max(), ce_a.max()) + pad
        y_min, y_max = min(orig_b.min(), ce_b.min()) - pad, max(orig_b.max(), ce_b.max()) + pad

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
        
        # El modelo requiere todas las variables, las demás se fijan a la media
        grid_full = np.zeros((xx.size, len(nombres_modelo)))
        for j, col in enumerate(nombres_modelo):
            if col == var_a: grid_full[:, j] = xx.ravel()
            elif col == var_b: grid_full[:, j] = yy.ravel()
            else: grid_full[:, j] = df[col].mean() if col in df.columns else 0.0

        grid_df = pd.DataFrame(grid_full, columns=nombres_modelo)
        Z_labels = modelo.predict(grid_df)
        
        clases_grid = sorted(list(set(clases_all) | set(Z_labels)))
        dict_clases = {cls: i for i, cls in enumerate(clases_grid)}
        Z_idx = np.array([dict_clases[val] for val in Z_labels]).reshape(xx.shape)
        
        plt.contourf(xx, yy, Z_idx, alpha=0.15, cmap=ListedColormap(colores_base[:len(clases_grid)]), 
                     levels=np.arange(len(clases_grid) + 1) - 0.5)
    else:
        clases_grid = clases_all

    # Dibujar puntos originales
    for cls in clases_all:
        idx_color = clases_grid.index(cls) if cls in clases_grid else clases_all.index(cls)
        plt.scatter(orig_a[df["pred_orig"] == cls], orig_b[df["pred_orig"] == cls], 
                    color=colores_base[idx_color % len(colores_base)], s=40, alpha=0.8, label=f"{cls}", zorder=3)

    # Dibujar flechas
    if FLECHAS:
        for xa, ya, xb, yb in zip(orig_a, orig_b, ce_a, ce_b):
            plt.annotate("", xy=(xb, yb), xytext=(xa, ya),
                         arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8, alpha=0.5))

    # Dibujar contraejemplos
    for cls in clases_all:
        idx_color = clases_grid.index(cls) if cls in clases_grid else clases_all.index(cls)
        mask = df["pred_orig"] == cls
        if mask.any():
            plt.scatter(ce_a[mask], ce_b[mask], color=colores_osc[idx_color % len(colores_osc)], 
                        marker="X", s=80, linewidths=0.6, edgecolors="white", label=f"Contraej. ({cls})", zorder=5)

    plt.xlabel(var_a); plt.ylabel(var_b)
    plt.title(f"Originales vs Contraejemplos  —  '{var_a}' y '{var_b}'")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    analizar_csv(CSV_PATH)
