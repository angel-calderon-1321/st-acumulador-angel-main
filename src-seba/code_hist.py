import numpy as np
import os
import torch
import pandas as pd
from autoencoder import NNAutoencoder
from read_data import read_raw, read_and_perform, train_test_split, scalings
import matplotlib.pyplot as plt
import seaborn as sns

def mean_absolute_error(sample, sample_pred):
    # Asegúrate de que ambos arrays sean de tipo numérico y no contengan NaNs o Infs
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    sample_pred = np.nan_to_num(sample_pred, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calcula el MAE
    mae = np.mean(np.abs(sample - sample_pred))
    return mae
#------------------------------------------------------------------------------------
# Leemos los archivos raw
folder = os.path.join('..', 'Date')
dataframes1 = read_raw(folder)
folder = os.path.join('..', 'Date2')
dataframes2 = read_raw(folder)

# Convertimos todo en arrays de numpy con series del mismo largo
data_1 = read_and_perform(dataframes1, row_range=300, col_range=(3,12), split= True)
data_2 = read_and_perform(dataframes2, row_range=99, col_range=(2,5), split= False)
# concatenamos todas las series
data_total = np.vstack([data_1.T, data_2.T])

# separamos train y test
train, test = train_test_split(data_total)

# escalamos
scaler = scalings(train)
train = scaler.fit_transform(train)
print(f'[+] Train shape {train.shape}')
test = scaler.transform(test)
print(f'[+] Test shape {test.shape}')

# Crear el directorio para guardar los modelos si no existe
path_model = os.path.join("..","Save_Models")
os.makedirs(path_model, exist_ok=True)

#verificamos que el path sea correcto y en caso que asi sea, vemos los nombres de archivos dentro
if os.path.isdir(os.path.join("..","Save_Models")):
    filename = os.listdir(os.path.join("..","Save_Models")) # guardamos la lista
    #print(filename)

dicc_model ={}
for f in filename:
    indl=9
    indr=f.find('.pth')
    #print(int(f[indl:indr]))
    dicc_model[int(f[indl:indr])] = f

# Crear el directorio para guardar las imágenes MAES si no existe
path_maes = os.path.join("..","img","MAES")
os.makedirs(path_maes, exist_ok=True)

mean_list = [] #media de cada esp. lat
std_list = [] #desviacion de cada esp. lat
dr = 0.2

for lat in range(1,21):
    for i in range(5):
        MAES_list = []
        autoencoder = NNAutoencoder(99, lat, dr)
        autoencoder.load_state_dict(torch.load(os.path.join("..","Save_Models",dicc_model[lat])))

        for i in range(500):
            autoencoder.train()
            x = torch.FloatTensor(test)
            y_pred = autoencoder(x)
            sample = scaler.inverse_transform(x)
            sample_pred = scaler.inverse_transform(y_pred.detach().numpy())
            #print(sample.shape, sample_pred.shape)
            average_MAES = mean_absolute_error(sample, sample_pred)
            MAES_list.append(average_MAES)
            print(f'DIM= {i} & Average MAE= {average_MAES:.3f}')

    array = np.array(MAES_list)
    mean_list.append(np.mean(array))
    std_list.append(np.std(array))

    # Crear el histograma
    #plt.figure(figsize=(10, 6))
    sns.histplot(MAES_list, kde = True, bins=50, edgecolor='black')
    plt.xlabel('MAES')
    plt.ylabel('f')
    plt.title(f'MAES, lat{lat} drop {dr}')
    plt.title('Training Loss')
    info_text = (
        f'mean: {np.mean(array):.3f}\n'
        f'std: {np.std(array):.3f}'
        )
    plt.text(0.70, 0.95, info_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top',horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightgray'))
    plt.savefig(os.path.join(path_maes,f"MAES{lat}-drop{dr}.png"), bbox_inches='tight')
    plt.close()

#lat vs MAES
latent = np.linspace(1,20,20)
fig, ax = plt.subplots(figsize = (10,6))
ax.errorbar(latent,mean_list,std_list,fmt='o', linewidth=2, capsize=6)
ax.plot(latent,mean_list)
ax.set_ylabel("MAES")
ax.set_xlabel("Laten Dim")
ax.set_xticks(np.arange(0, 22, 1))
ax.grid()
plt.savefig(os.path.join(path_maes,f"MAES-LAT.png"), bbox_inches='tight')
plt.close()