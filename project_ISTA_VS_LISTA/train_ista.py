from ista import ISTA
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ista import ISTA
import sys
sys.path.append(Path(__file__).resolve().parent.parent) 

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
K = 1000
X_init = np.zeros_like(x_true)

# Initialisation
model_ista = ISTA(A=A, b=b_test, L=L, lag=lam, x_true=x_true)

X_ista, nmse_history_ista = model_ista.forward(X_init, epochs=K)

np.savez("project_ISTA_VS_LISTA/resultats_train/resultats_ista.npz", X_ista=X_ista, nmse_history_ista=nmse_history_ista)
print("ISTA training completed. Results saved to 'resultats_ista.npz'.")