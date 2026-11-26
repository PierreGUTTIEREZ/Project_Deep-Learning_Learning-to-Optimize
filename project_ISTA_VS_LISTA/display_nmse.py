import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ista import ISTA
from fista import FISTA

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

# Initialisation
model_ista = ISTA(A=A, b=b_test, L=L, lag=lam, x_true=x_true)
model_fista = FISTA(A=A, b=b_test, L=L, lag=lam, x_true=x_true) 
X_init = np.zeros_like(x_true)

epochs = 1000
print(f"Exécution d'ISTA sur {epochs} itérations...")
X_ista, nmse_history_ista = model_ista.forward(X_init, epochs=epochs)
X_fista, nmse_history_fista = model_fista.forward(X_init, epochs=epochs)

# Affichage graphique
plt.figure(figsize=(8, 5))
plt.plot(range(len(nmse_history_ista)), nmse_history_ista, label="ISTA", color="navy", lw=2)
plt.plot(range(len(nmse_history_fista)), nmse_history_fista, label="FISTA", color="red", lw=2)
plt.xlabel("Itérations")
plt.ylabel("NMSE (dB)")
plt.title(f"Convergence d'ISTA sur le LASSO (noiseless {A.shape[0]}x{A.shape[1]})")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()