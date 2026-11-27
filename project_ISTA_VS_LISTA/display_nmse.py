import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from fista import FISTA
import sys
import torch
sys.path.append(Path(__file__).resolve().parent.parent)   # pour importer lista.py (à la racine du projet)
from lista import LISTA
from lfista import LFISTA

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "noiseless_256_512"

A = np.load(data_dir / "A.npy")
data = np.load(data_dir / "test_data.npy")

M = A.shape[0]  # 256
b_test = data[:, :M]       # Shape: (1024, 256)
x_true = data[:, M:]       # Shape: (1024, 512)


L = np.linalg.svd(A, compute_uv=False)[0]**2
print(f"Dimensions A: {A.shape}, L = {L:.4f}")

lam = 0.005

K = 1000   # doit être le même que dans train_lista.py (lam aussi)

# importation des résultats d'ISTA
data = np.load("project_ISTA_VS_LISTA/resultats_train/resultats_ista.npz")
X_ista = data["X_ista"]
nmse_history_ista = data["nmse_history_ista"]

# importation des résultats de FISTA
data = np.load("project_ISTA_VS_LISTA/resultats_train/resultats_fista.npz")
X_fista = data["X_fista"]
nmse_history_fista = data["nmse_history_fista"]

# importation des résultats de LISTA
model_lista = LISTA(A=A, L=L, lag=lam, K=K)
model_lista.load_state_dict(torch.load(BASE_DIR / "project_ISTA_VS_LISTA" / "resultats_train" / f"lista_K{K}.pt"))
model_lista.eval()

with torch.no_grad():
    x_layers = model_lista(torch.from_numpy(b_test), return_all=True)
nmse_lista = [0.0] + [model_lista.compute_nmse_db(x_k, torch.from_numpy(x_true)) for x_k in x_layers]

# importation des résultats de LFISTA
model_lfista = LFISTA(A=A, L=L, lag=lam, K=K)
model_lista.load_state_dict(torch.load(BASE_DIR / "project_ISTA_VS_LISTA" / "resultats_train" / f"lfista_K{K}.pt"))
model_lista.eval()

with torch.no_grad():
    x_layers = model_lfista(torch.from_numpy(b_test), return_all=True)
nmse_lfista = [0.0] + [model_lfista.compute_nmse_db(x_k, torch.from_numpy(x_true)) for x_k in x_layers]

# Affichage graphique
plt.figure(figsize=(8, 5))
plt.plot(range(len(nmse_history_ista)), nmse_history_ista, label="ISTA", color="navy", lw=2)
plt.plot(range(len(nmse_history_fista)), nmse_history_fista, label="FISTA", color="red", lw=2)
plt.plot(range(K + 1), nmse_lista, "-", label=f"LISTA ({K} couches)", color="darkorange", lw=2)
plt.plot(range(K + 1), nmse_lfista, "-", label=f"LFISTA ({K} couches)", color="chartreuse", lw=2)
plt.xlabel("Itérations")
plt.ylabel("NMSE (dB)")
plt.title(f"Convergence d'ISTA sur le LASSO (noiseless {A.shape[0]}x{A.shape[1]})")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()