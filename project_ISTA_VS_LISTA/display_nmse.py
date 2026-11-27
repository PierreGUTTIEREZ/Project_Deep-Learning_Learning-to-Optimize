import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ista import ISTA
from fista import FISTA
import sys
import torch
sys.path.append(Path(__file__).resolve().parent.parent)   # pour importer lista.py (à la racine du projet)
from lista import LISTA

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

K = 16   # doit être le même que dans train_lista.py (lam aussi)

# Initialisation
model_ista = ISTA(A=A, b=b_test, L=L, lag=lam, x_true=x_true)
model_fista = FISTA(A=A, b=b_test, L=L, lag=lam, x_true=x_true) 
# LISTA
model_lista = LISTA(A=A, L=L, lag=lam, K=K)
model_lista.load_state_dict(torch.load(BASE_DIR / "project_ISTA_VS_LISTA" / "resultats_train" / f"lista_K{K}.pt"))
model_lista.eval()

X_init = np.zeros_like(x_true)

epochs = 1000
print(f"Exécution d'ISTA sur {epochs} itérations...")
X_ista, nmse_history_ista = model_ista.forward(X_init, epochs=epochs)
X_fista, nmse_history_fista = model_fista.forward(X_init, epochs=epochs)
# LISTA
with torch.no_grad():
    x_layers = model_lista(torch.from_numpy(b_test), return_all=True)
nmse_lista = [0.0] + [model_lista.compute_nmse_db(x_k, torch.from_numpy(x_true)) for x_k in x_layers]  # 0 dB pour x^(0) = 0


# Affichage graphique
plt.figure(figsize=(8, 5))
plt.plot(range(len(nmse_history_ista)), nmse_history_ista, label="ISTA", color="navy", lw=2)
plt.plot(range(len(nmse_history_fista)), nmse_history_fista, label="FISTA", color="red", lw=2)
plt.plot(range(K + 1), nmse_lista, "-", label=f"LISTA ({K} couches)", color="darkorange", lw=2)
plt.xlabel("Itérations")
plt.ylabel("NMSE (dB)")
plt.title(f"Convergence d'ISTA sur le LASSO (noiseless {A.shape[0]}x{A.shape[1]})")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()