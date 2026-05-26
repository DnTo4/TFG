import pandas as pd
import matplotlib.pyplot as plt

"""Utilidades gráficas para la representación simple de datasets bidimensionales.

Ofrece funciones para cargar datasets, validar columnas y mostrar gráficos.
"""

def graficar_datos(path_archivo, columna_x, columna_y, columna_clase):
    """Cargar y graficar las instancias de un dataset.

    Lee el archivo CSV, comprueba que las columnas existen en el DataFrame
    y dibuja las clases usando scatter plots.
    """
    try:
        # Cargar el dataset
        df = pd.read_csv(path_archivo)
        
        # Validar la existencia de las columnas
        for col in [columna_x, columna_y, columna_clase]:
            if col not in df.columns:
                print(f"Error: La columna '{col}' no se encuentra en el archivo.")
                return

        # Configurar las dimensiones de la figura
        plt.figure(figsize=(10, 7))
        
        # Obtener los identificadores de clase
        clases = df[columna_clase].unique()
        
        # Graficar cada una de las clases
        for clase in clases:
            mask = df[columna_clase] == clase
            plt.scatter(
                df.loc[mask, columna_x], 
                df.loc[mask, columna_y], 
                label=clase,
                alpha=0.8,
                edgecolors='white',
                s=70
            )
            
        plt.title(f'{columna_x} vs {columna_y}', fontsize=14, fontweight='bold')
        plt.xlabel(columna_x, fontsize=12)
        plt.ylabel(columna_y, fontsize=12)
        plt.legend(title=columna_clase, loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Ajustar dimensiones y mostrar el gráfico
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {path_archivo}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

def main():
    """Ejecutar la visualización a partir de argumentos de consola."""
    import argparse
    parser = argparse.ArgumentParser(description="Grafica dos variables de un dataset coloreando por clase.")
    parser.add_argument("--archivo", type=str, default="datos/originales/train_moons.csv", help="Ruta del archivo CSV")
    parser.add_argument("--eje-x", type=str, default="x1", help="Nombre de la columna para el eje X")
    parser.add_argument("--eje-y", type=str, default="x2", help="Nombre de la columna para el eje Y")
    parser.add_argument("--clase", type=str, default="y", help="Nombre de la columna de clase")
    args = parser.parse_args()

    print(f"--- Iniciando visualización de {args.archivo} ---")
    
    # Visualización de datos
    graficar_datos(
        args.archivo, 
        args.eje_x, 
        args.eje_y, 
        args.clase
    )

if __name__ == "__main__":
    main()