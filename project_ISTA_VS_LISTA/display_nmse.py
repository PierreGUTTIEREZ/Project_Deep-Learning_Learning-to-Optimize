import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from project_ISTA_VS_LISTA.models.lista import LISTA
from project_ISTA_VS_LISTA.models.lfista import LFISTA
from project_ISTA_VS_LISTA.models.lista_cp import LISTA_CP

# Gestion automatique GPU / CPU compatible multi-utilisateurs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Périphérique sélectionné : {device}")

# Chemins
data_dir = BASE_DIR / "data" / "noiseless_256_512"
res_dir = BASE_DIR / "project_ISTA_VS_LISTA" / "resultats_train"

A = np.load(data_dir / "A.npy")
data = np.load(data_dir / "test_data.npy")

M = A.shape[0]  # 256
b_test = torch.from_numpy(data[:, :M]).float().to(device)
x_true = torch.from_numpy(data[:, M:]).float().to(device)

L = np.linalg.svd(A, compute_uv=False)[0]**2
print(f"Dimensions A: {A.shape}, L = {L:.4f}")

lam = 0.005
K = 16

# 1. Importation des résultats d'ISTA
data_ista = np.load(res_dir / "resultats_ista.npz")
nmse_history_ista = data_ista["nmse_history_ista"]

# 2. Importation des résultats de FISTA
data_fista = np.load(res_dir / "resultats_fista.npz")
nmse_history_fista = data_fista["nmse_history_fista"]

# 3. Importation des résultats de LISTA
p_lista = res_dir / f"lista_K{K}.pt"
nmse_lista = None
if p_lista.exists():
    model_lista = LISTA(A=A, L=L, lag=lam, K=K).to(device)
    model_lista.load_state_dict(torch.load(p_lista, map_location=device, weights_only=False))
    model_lista.eval()
    with torch.no_grad():
        x_layers = model_lista(b_test, return_all=True)
    nmse_lista = [0.0] + [model_lista.compute_nmse_db(x_k, x_true) for x_k in x_layers]

# 4. Importation des résultats de LFISTA
p_lfista = res_dir / f"lfista_K{K}.pt"
nmse_lfista = None
if p_lfista.exists():
    model_lfista = LFISTA(A=A, L=L, lag=lam, K=K).to(device)
    model_lfista.load_state_dict(torch.load(p_lfista, map_location=device, weights_only=False))
    model_lfista.eval()
    with torch.no_grad():
        x_layers = model_lfista(b_test, return_all=True)
    nmse_lfista = [0.0] + [model_lfista.compute_nmse_db(x_k, x_true) for x_k in x_layers]

# 5. Importation des résultats de LISTA-CP (Coupled Parameters)
p_listacp = res_dir / f"lista_cp_K{K}.pt"
nmse_lista_cp = None
if p_listacp.exists():
    model_lista_cp = LISTA_CP(A=A, L=L, lag=lam, K=K).to(device)
    model_lista_cp.load_state_dict(torch.load(p_listacp, map_location=device, weights_only=False))
    model_lista_cp.eval()
    with torch.no_grad():
        x_layers_cp = model_lista_cp(b_test, return_all=True)
    nmse_lista_cp = [0.0] + [model_lista_cp.compute_nmse_db(x_k, x_true) for x_k in x_layers_cp]

# Affichage graphique
plt.figure(figsize=(9, 5.5))
plt.plot(range(K + 1), nmse_history_ista[:K + 1], label="ISTA", color="navy", lw=2)
plt.plot(range(K + 1), nmse_history_fista[:K + 1], label="FISTA", color="red", lw=2)
if nmse_lista is not None:
    plt.plot(range(K + 1), nmse_lista, "-", label=f"LISTA ({K} couches)", color="darkorange", lw=2)
if nmse_lfista is not None:
    plt.plot(range(K + 1), nmse_lfista, "-", label=f"LFISTA ({K} couches)", color="chartreuse", lw=2)
if nmse_lista_cp is not None:
    plt.plot(range(K + 1), nmse_lista_cp, "--", label=f"LISTA-CP ({K} couches)", color="magenta", lw=2.5)

plt.xlabel("Itérations / Couches")
plt.ylabel("NMSE (dB)")
plt.title(f"Convergence sur le LASSO (noiseless {A.shape[0]}x{A.shape[1]}) pour K={K}")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

save_path = res_dir / f"comparaison_nmse_K{K}.png"
plt.savefig(save_path, dpi=300)
print(f"Graphique sauvegardé dans : {save_path}")

if hasattr(sys, 'ps1') or not sys.flags.interactive:
    plt.close()
else:
    plt.show()
