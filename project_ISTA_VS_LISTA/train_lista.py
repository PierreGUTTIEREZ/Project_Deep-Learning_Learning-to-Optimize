import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))                          # pour importer lista.py (à la racine du projet)
sys.path.append(str(Path(__file__).resolve().parent))   # pour importer ista.py (dans ce dossier)
from lista import LISTA

data_dir = BASE_DIR / "data" / "noiseless_256_512"
torch.manual_seed(1)

# Données : une ligne = [b, x*]
A = np.load(data_dir / "A.npy")
train_data = np.load(data_dir / "train_data.npy")
val_data = np.load(data_dir / "val_data.npy")
test_data = np.load(data_dir / "test_data.npy")

M = A.shape[0]  # 256
b_train, x_train = torch.from_numpy(train_data[:, :M]), torch.from_numpy(train_data[:, M:])  # (51200, 256), (51200, 512)
b_val, x_val = torch.from_numpy(val_data[:, :M]), torch.from_numpy(val_data[:, M:])          # (1024, 256), (1024, 512)
b_test, x_test = torch.from_numpy(test_data[:, :M]), torch.from_numpy(test_data[:, M:])      # (1024, 256), (1024, 512)

L = np.linalg.svd(A, compute_uv=False)[0]**2
print(f"Dimensions A: {A.shape}, L = {L:.4f}")

lam = 0.005
K = 16

# Création du réseau
model_lista = LISTA(A=A, L=L, lag=lam, K=K)
print(f"Nombre de paramètres : {sum(p.numel() for p in model_lista.parameters()):,}")

# Entraînement

# Loss function: MSE entre x estimé et x*
criterion = nn.MSELoss()

# Optimizer: Adam (SGD est beaucoup trop lent ici), lr de l'article
optimizer = torch.optim.Adam(model_lista.parameters(), lr=5e-4)

# Le lr est divisé par 2 toutes les 5 epochs (LISTA devient instable avec un lr trop grand en fin d'entraînement)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Mini-batchs de 128 exemples (tout le train d'un coup ne tiendrait pas en mémoire)
train_loader = DataLoader(TensorDataset(b_train, x_train), batch_size=128, shuffle=True)

epochs = 200 # number of epochs
losses = [] # loss moyenne sur chaque epoch
nmse_val = [] # NMSE sur la validation à chaque epoch
best_nmse = float("inf") # meilleure NMSE de validation rencontrée
best_weights = None # poids correspondants

# Loop on epochs
for i in range(epochs):

    loss_epoch = 0.0

    # Loop on mini-batchs
    for b_batch, x_batch in train_loader:

        # compute the prediction using the previous parameters of the neural network
        x_pred = model_lista(b_batch)

        # compute the loss
        loss = criterion(x_pred, x_batch)
        loss_epoch += loss.item()

        # initialize the gradient to zero
        optimizer.zero_grad()

        # compute the gradient by back propagation
        loss.backward()

        # limite la norme du gradient (évite qu'un batch "explose" et fasse diverger le réseau)
        torch.nn.utils.clip_grad_norm_(model_lista.parameters(), max_norm=1.0)

        # update the parameter values using the gradient
        optimizer.step()

    scheduler.step()
    losses.append(loss_epoch / len(train_loader))

    # NMSE sur la validation (pas besoin de gradient ici)
    with torch.no_grad():
        nmse_val.append(model_lista.compute_nmse_db(model_lista(b_val), x_val))

    # on garde les poids qui donnent la meilleure NMSE de validation
    if nmse_val[-1] < best_nmse:
        best_nmse = nmse_val[-1]
        best_weights = {name: w.clone() for name, w in model_lista.state_dict().items()}

    print(f"Epoch {i+1:3d}/{epochs} | loss = {losses[-1]:.2e} | NMSE val = {nmse_val[-1]:.2f} dB")

# On recharge le meilleur modèle et on le sauvegarde
model_lista.load_state_dict(best_weights)
print(f"Meilleure NMSE de validation : {best_nmse:.2f} dB")
models_dir = BASE_DIR / "project_ISTA_VS_LISTA" / "resultats_train"
models_dir.mkdir(parents=True, exist_ok=True)
torch.save(model_lista.state_dict(), models_dir / f"lista_K{K}.pt")