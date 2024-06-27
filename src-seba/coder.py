import numpy as np
import os
import torch
import pandas as pd
from autoencoder import NNAutoencoder
from read_data import read_raw, read_and_perform, train_test_split, scalings
import matplotlib.pyplot as plt
import seaborn as sns

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

epochs = 1000
lr = 1e-3
dr = 0.2
lat = 2

# Crear el directorio para guardar las imágenes si no existe
path_img = os.path.join("..","img","Best_Model")
os.makedirs(path_img, exist_ok=True)

# Entrenamiento
hist_train = []
hist_test = []
best_test_loss = 100
best_epoch = 0
best_model = None
espera = 100
b = 0 #bandera

autoencoder = NNAutoencoder(99, lat, dr)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr = lr)
criterio = torch.nn.MSELoss()

for e in range(epochs):
    autoencoder.train()
    x = torch.FloatTensor(train)
    y_pred = autoencoder(x)
    loss = criterio(y_pred, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    hist_train.append(loss.item())
    
    with torch.no_grad():
        autoencoder.eval()
        x = torch.FloatTensor(test)
        y_pred = autoencoder(x)
        test_loss = criterio(y_pred, x)
        hist_test.append(test_loss.item())

    if e%100 == 0:
            print(f'Epoch {e}, train Loss: {loss.item():.4f}, test Loss: {test_loss.item():.4f}')

    if test_loss < loss:
        #guardo el mejor modelo
        if test_loss < best_test_loss: #encuantra el primer minimo??
            best_test_loss = test_loss
            best_epoch = e

        if (e - best_epoch >= espera) and (b < 3):
            m_epoch = best_epoch
            m_test_loss = best_test_loss
            best_model = autoencoder.state_dict().copy() #copia del mejor modelo
            torch.save(best_model, os.path.join(path_model,f"model-lat{lat}.pth"))
            b += 1
            plt.plot(m_epoch,m_test_loss,'x', color = "red")

print(f'Best epoch was {m_epoch} with val loss {m_test_loss:.4f}')
#guardo las img
plt.semilogy(hist_train, label = 'train loss')
plt.semilogy(hist_test, label = 'test loss')
plt.plot(m_epoch,m_test_loss,'x', color = "red", label = "best model")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title(f"dim_lat: {lat}, drop = {dr}")
plt.legend()
plt.savefig(os.path.join(path_img,f"BestModel-lat{lat}-drop{dr}.png"), bbox_inches='tight')
plt.close()