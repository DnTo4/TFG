import os
import sys
import argparse
import subprocess
import time

def run_script(script_name, args_list):
    print(f"Ejecutando {script_name}...")
    t_start = time.time()
    
    cmd = [sys.executable, script_name] + args_list
    result = subprocess.run(cmd)
    
    t_elapsed = time.time() - t_start
    if result.returncode == 0:
        print(f"Completado {script_name} en {t_elapsed:.2f}s.\n")
        return True, t_elapsed
    else:
        print(f"Error en {script_name} (código de salida: {result.returncode}) en {t_elapsed:.2f}s.\n")
        return False, t_elapsed

def main():
    parser = argparse.ArgumentParser(description="Programa principal - Ejecución de flujo de TFG")
    parser.add_argument("--modelo", type=str, default="svm", choices=["svm", "mlp"], help="Modelo base (svm, mlp)")
    parser.add_argument("--dataset", type=str, default="datos/originales/diabetes.csv", help="Dataset principal")
    parser.add_argument("--hiperparams", action="store_true", help="Habilitar el estudio de hiperparámetros (muy lento)")
    args = parser.parse_args()

    print("Inicio de flujo principal de TFG")
    print(f"Configuración: Modelo = {args.modelo}, Dataset = {args.dataset}, Hiperparámetros = {args.hiperparams}\n")
    
    t_total_start = time.time()
    
    pipeline = []
    
    # Validación con datos sintéticos
    pipeline.append(("validar_sinteticos.py", ["--modelo", args.modelo, "--dataset", "datos/originales/train_moons.csv"]))
    
    # Estudio de hiperparámetros
    if args.hiperparams:
        pipeline.append(("estudio_hiperparametros.py", []))
        
    # Comparativa de generadores
    pipeline.append(("comparar_generadores.py", ["--modelo", args.modelo, "--dataset", args.dataset, "--algoritmo", "todos"]))
    
    # Evaluación de realismo de contraejemplos
    pipeline.append(("evaluar_realismo.py", ["--dataset", args.dataset, "--input", "datos/procesados/contraejemplos_memetico.csv", "--modelo", "modelos/modelo.joblib"]))
    
    # Análisis de rendimiento y escalabilidad de algoritmos
    pipeline.append(("analisis_rendimiento_algoritmos.py", ["--modelo", args.modelo, "--dataset", "datos/originales/train_moons.csv"]))
    
    # Simplificación del modelo
    pipeline.append(("simplificar_modelo.py", ["--modelo", args.modelo, "--dataset", args.dataset]))
    
    # Análisis de deriva espacial
    pipeline.append(("deriva.py", ["--modelo", args.modelo, "--dataset", "datos/originales/iris.data"]))

    tiempos = {}
    
    for i, (script, script_args) in enumerate(pipeline):
        success, duration = run_script(script, script_args)
        tiempos[script] = duration
        if not success:
            print("Ejecución cancelada por fallo en etapa crítica.")
            sys.exit(1)
            
        if i < len(pipeline) - 1:
            input("Presiona Intro para continuar con la siguiente etapa...")
            print()

            
    t_total = time.time() - t_total_start
    print("Resumen de tiempos de ejecución:")
    for script in tiempos:
        print(f"  - {script}: {tiempos[script]:.2f}s")
    print(f"Tiempo total transcurrido: {t_total:.2f}s")

if __name__ == "__main__":
    main()
