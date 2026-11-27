import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))
from project_ISTA_VS_LISTA.models.lista_cp import LISTA_CP

data_dir = BASE_DIR / "data" / "noiseless_256_512"
torch.manual_seed(1)

# Chargement des données
A = np.load(data_dir / "A.npy")
train_data = np.load(data_dir / "train_data.npy")
val_data = np.load(data_dir / "val_data.npy")
test_data = np.load(data_dir / "test_data.npy")

M = A.shape[0]  # 256
b_train, x_train = torch.from_numpy(train_data[:, :M]), torch.from_numpy(train_data[:, M:])
b_val, x_val = torch.from_numpy(val_data[:, :M]), torch.from_numpy(val_data[:, M:])
b_test, x_test = torch.from_numpy(test_data[:, :M]), torch.from_numpy(test_data[:, M:])

L = np.linalg.svd(A, compute_uv=False)[0]**2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dimensions A: {A.shape}, L = {L:.4f} | Device: {device}")

lam = 0.005
K = 16

# Création du réseau LISTA-CP
model_lista_cp = LISTA_CP(A=A, L=L, lag=lam, K=K).to(device)
b_val, x_val = b_val.to(device), x_val.to(device)
print(f"Nombre de paramètres LISTA-CP (K={K}) : {sum(p.numel() for p in model_lista_cp.parameters()):,}")

# Entraînement
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_lista_cp.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
train_loader = DataLoader(TensorDataset(b_train, x_train), batch_size=128, shuffle=True)

epochs = 20
losses = []
nmse_val = []
best_nmse = float("inf")
best_weights = None

print(f"Début de l'entraînement de LISTA-CP sur {epochs} époques...")
for i in range(epochs):
    loss_epoch = 0.0

    for b_batch, x_batch in train_loader:
        b_batch, x_batch = b_batch.to(device), x_batch.to(device)

        x_pred = model_lista_cp(b_batch)
        loss = criterion(x_pred, x_batch)
        loss_epoch += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_lista_cp.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()
    losses.append(loss_epoch / len(train_loader))

    with torch.no_grad():
        nmse_val.append(model_lista_cp.compute_nmse_db(model_lista_cp(b_val), x_val))

    if nmse_val[-1] < best_nmse:
        best_nmse = nmse_val[-1]
        best_weights = {name: w.clone() for name, w in model_lista_cp.state_dict().items()}

    print(f"Epoch {i+1:3d}/{epochs} | loss = {losses[-1]:.2e} | NMSE val = {nmse_val[-1]:.2f} dB")

# Sauvegarde du meilleur modèle
model_lista_cp.load_state_dict(best_weights)
print(f"Meilleure NMSE de validation LISTA-CP : {best_nmse:.2f} dB")
models_dir = BASE_DIR / "project_ISTA_VS_LISTA" / "resultats_train"
models_dir.mkdir(parents=True, exist_ok=True)
torch.save(model_lista_cp.state_dict(), models_dir / f"lista_cp_K{K}.pt")
print(f"Modèle sauvegardé dans {models_dir / f'lista_cp_K{K}.pt'}")
