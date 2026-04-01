import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path, target_column=None):
    # Leer datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Si no se indica variable objetivo → usar la última columna
    if target_column is None:
        target_column = df_train.columns[-1]

    # Separar variables
    y_train = df_train[target_column]
    X_train = df_train.drop(columns=[target_column])
    
    y_test = df_test[target_column]
    X_test = df_test.drop(columns=[target_column])

    # Convertir variables categóricas
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined = pd.get_dummies(X_combined)

    X_train = X_combined.iloc[:len(X_train)]
    X_test = X_combined.iloc[len(X_train):]

    return X_train, y_train, X_test, y_test

def train_model(train_path, test_path, target_column=None):
    # Cargar datos
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

    # Crear pipeline SVM
    modelo = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
    )

    # Entrenar
    modelo.fit(X_train, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Guardar nombres de columnas
    nombres = X_train.columns.tolist()

    # Devolver DataFrames para mantener compatibilidad con pipeline
    return modelo, (X_train, y_train, X_test, y_test), acc, nombres
