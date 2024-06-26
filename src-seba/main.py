import numpy as np
import os
import torch
import pandas as pd
from autoencoder import NNAutoencoder
from read_data import read_raw, read_and_perform, train_test_split, scalings
import matplotlib.pyplot as plt



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
test = scaler.fit_transform(test)
print(f'[+] Test shape {test.shape}')


# el modelo
autoencoder = NNAutoencoder(99, 1, 0.5)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr = 1e-3)
criterio = torch.nn.MSELoss()
# Entrenamiento
hist_train = []
hist_test = []
for e in range(3000):
    autoencoder.train()
    x = torch.FloatTensor(train)
    y_pred = autoencoder(x)
    loss = criterio(y_pred, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e%100 == 0:
        print(e, loss.item())
    hist_train.append(loss.item())
    with torch.no_grad():
        autoencoder.eval()
        x = torch.FloatTensor(test)
        y_pred = autoencoder(x)
        loss = criterio(y_pred, x)
        hist_test.append(loss.item())


plt.semilogy(hist_train, label = 'train loss')
plt.semilogy(hist_test, label = 'test loss')
plt.legend()
plt.show()

# plot a single sample
with torch.no_grad():
    autoencoder.eval()
    x = torch.FloatTensor(test[10]).view(1,99)
    y_pred = autoencoder(x)
    sample = scaler.inverse_transform(test[10].reshape(1,99)).reshape(-1)
    sample_pred = scaler.inverse_transform(y_pred.numpy().reshape(1, 99)).reshape(-1)

RMSE = np.sqrt(((sample-sample_pred)**2).mean())
plt.plot(sample, label = 'ground truth')
plt.plot(sample_pred, label = 'model prediction')
plt.text(10,20, f'RMSE: {RMSE:.3f}  Celsius')
plt.legend()
plt.show()


