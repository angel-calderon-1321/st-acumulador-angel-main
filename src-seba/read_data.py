import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def read_raw(folder):
    '''ojo que estoy llenando los nan con ceros'''
    dataframes = []
    for f in os.listdir(folder):
        dataframes.append(pd.read_excel(os.path.join(folder, f)).fillna(0))
    return dataframes

def read_and_perform(dfs, row_range = 300, col_range=(3,12), split = False):
    '''
    Esta funcion lee los dataframes y devuelve un array de numpy con 99 filas
    y tantas columnas como series de mediciones
    '''
    n_data = 99
    # extraemos de los dataframes los arrays de numpy y los acumulamos en una lista
    data = []
    for d in dfs:
        data.append(d.values[:row_range,col_range[0]:col_range[1]])

    # si los dataframes tienen mas de 99 datos dividimos la serie en dos
    # tomando los maximos y promedios
    nw = row_range//n_data
    series = np.zeros(shape=(99,1))
    if split:
        for d in data:
            # los maximos en cada ventanita de reduccion (MAXPOOL)
            max_serie = np.array([d[nw*i:(i+1)*nw,:].max(axis=0) for i in range(len(d)//nw)])
            max_serie = max_serie[:-1,:]

            #AVERAGEPOOL
            avg_serie = np.array([d[nw*i:(i+1)*nw,:].mean(axis=0) for i in range(len(d)//nw)])
            avg_serie = avg_serie[:-1,:]

            #STRIDE
            first_serie = d[::nw,:]
            first_serie = first_serie[:-1,:]

            series = np.hstack([series, first_serie])
            series = np.hstack([series,max_serie])
            series = np.hstack([series,avg_serie])
    else:
        for d in data:
            series = np.hstack([series, d])


    series = series[:,1:]
    print(f'[+] Se procesaron {series.shape[1]} series de longitud {n_data}')
    return series

def scalings(super_array):

    scaler = StandardScaler()
    scaler.fit(super_array.reshape(-1,1))
    print('[+] StandardScaler entrenado')

    return scaler

def scale_serie(x, scaler):
    return scaler.fit_transform(x)

def train_test_split(data, test_ratio = 0.2, shuffle = True):
    '''data tiene que ser un array cada columna una serie 
    devuelve (train, test)'''
    if shuffle:
        rng = np.random.default_rng()
        rng.shuffle(data, axis= 0)    

    n_test = np.int32(data.shape[1] * test_ratio)
    return data[n_test:], data[:n_test]
