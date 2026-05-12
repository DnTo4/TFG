import pandas as pd
import matplotlib.pyplot as plt

def graficar_datos(path_archivo, columna_x, columna_y, columna_clase):
    try:
        # 1. Cargar el dataset
        df = pd.read_csv(path_archivo)
        
        # Verificar que las columnas existan
        for col in [columna_x, columna_y, columna_clase]:
            if col not in df.columns:
                print(f"Error: La columna '{col}' no se encuentra en el archivo.")
                return

        # 2. Configurar la figura
        plt.figure(figsize=(10, 7))
        
        # 3. Obtener las clases únicas para asignar colores
        clases = df[columna_clase].unique()
        
        # 4. Graficar cada clase por separado
        # Esto permite que Matplotlib asigne colores distintos y cree la leyenda
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

        # 5. Personalización estética
        plt.title(f'{columna_x} vs {columna_y}', fontsize=14, fontweight='bold')
        plt.xlabel(columna_x, fontsize=12)
        plt.ylabel(columna_y, fontsize=12)
        plt.legend(title=columna_clase, loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 6. Mostrar el resultado
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {path_archivo}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

def main():
    # --- CONFIGURACIÓN DEL USUARIO ---
    # Reemplaza estos valores con los nombres de tu archivo y columnas
    config = {
        "archivo": "datos/originales/train_moons.csv",
        "eje_x": "x1",
        "eje_y": "x2",
        "clase": "y"
    }
    # ---------------------------------

    print(f"--- Iniciando visualización de {config['archivo']} ---")
    
    # Nota: Asegúrate de que el archivo existe o cambia la ruta.
    graficar_datos(
        config["archivo"], 
        config["eje_x"], 
        config["eje_y"], 
        config["clase"]
    )

if __name__ == "__main__":
    main()