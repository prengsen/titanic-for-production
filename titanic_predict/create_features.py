"""
    Este modulo crea las características de nuestro modelo.
"""
import pickle
import pandas as pd
import numpy as np

def create_model_features():
    """
    Esta función genera las features para el modelo de ML
    """
    dataset = pd.read_csv("../data/raw/train.csv")
    dataset.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    # eliminamos cabin por cantidad de faltantes.
    dataset.drop(['Cabin'], axis=1, inplace=True)

    # Tamaño del conjunto de prueba (por ejemplo, 20%)
    test_size = 0.2

    # Obtener el número de filas
    num_rows = len(dataset)

    # Crear índices aleatorios para el conjunto de prueba
    test_indices = np.random.choice(num_rows, int(num_rows * test_size), replace=False)

    # Crear un DataFrame booleano para indicar si una fila pertenece al conjunto de prueba
    is_test = np.isin(np.arange(num_rows), test_indices)

    # Dividir el DataFrame
    train = dataset.loc[~is_test]
    test = dataset.loc[is_test]

    # Imputamos la media en la columna Age
    media_age = train['Age'].mean()
    train['Age'] = train['Age'].fillna(media_age).astype(int)

    # Imputamos Embarked con valor mas frecuente.
    embarked_imputed_value = train['Embarked'].value_counts().index[0]
    train['Embarked'] = train['Embarked'].fillna(embarked_imputed_value)

    # Codificación de variable Sex con One Hot Encoding
    train['Sex'] = pd.get_dummies(train['Sex'], drop_first=True).astype(int)

    # Codificación de variable Embarked con Frecuency Encoding
    codificador_embarked = train['Embarked'].value_counts()
    train['Embarked'] = train['Embarked'].map(codificador_embarked)

# Gurdamos valores de configuración del train.
    feature_eng_configs = {
        'age_imputed_value': int(media_age),
        'embarked_imputed_value': embarked_imputed_value,
        'codificador_embarked': codificador_embarked
    }

    with open('../artifacts/feature_eng_configs.pkl', 'wb') as f:
        pickle.dump(feature_eng_configs, f)

    # Guardamos el Dataset Procesado.
    train.to_csv('../data/processed/features_for_model.csv', index=False)
    test.to_csv('../data/processed/test_dataset.csv', index=False)
#llamada a función para ajecución.
create_model_features()
