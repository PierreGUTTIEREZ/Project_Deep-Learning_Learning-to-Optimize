import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))   # pour importer les classes (dans ce dossier)
from ista import ISTA
from fista import FISTA
from lista import LISTA
from lfista import LFISTA

data_dir = BASE_DIR / "data" / "noiseless_256_512"
resultats_dir = BASE_DIR / "project_ISTA_VS_LISTA" / "resultats_train"

# ---------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------
lam = 0.005    # doit être le même que dans train_lista.py / train_lfista.py
K = 16         # nombre de couches de LISTA / LFISTA

# Temps d'entraînement réellement mesurés (en secondes) avec Measure-Command dans PowerShell.
# Laisser None pour utiliser l'estimation calculée par ce script.
T_TRAIN_LISTA_MESURE = None
T_TRAIN_LFISTA_MESURE = None

# Réglages de l'entraînement (les mêmes que dans train_lista.py / train_lfista.py)
epochs_train = 20
batch_size = 128

n_repet = 10        # chaque temps est mesuré n_repet fois, on garde la médiane
n_pas_estim = 50    # nombre de pas d'entraînement chronométrés pour l'estimation

print(f"Threads PyTorch : {torch.get_num_threads()}")

# ---------------------------------------------------------------
# Données et modèles
# ---------------------------------------------------------------
A = np.load(data_dir / "A.npy")
data = np.load(data_dir / "test_data.npy")
train_data = np.load(data_dir / "train_data.npy")

M = A.shape[0]  # 256
b_test = data[:, :M]       # Shape: (1024, 256)
x_true = data[:, M:]       # Shape: (1024, 512)
n_test = b_test.shape[0]
n_train = train_data.shape[0]

L = np.linalg.svd(A, compute_uv=False)[0]**2
print(f"Dimensions A: {A.shape}, L = {L:.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_lista = LISTA(A=A, L=L, lag=lam, K=K).to(device)
model_lista.load_state_dict(torch.load(resultats_dir / f"lista_K{K}.pt", map_location=device, weights_only=False))
model_lista.eval()

model_lfista = LFISTA(A=A, L=L, lag=lam, K=K).to(device)
model_lfista.load_state_dict(torch.load(resultats_dir / f"lfista_K{K}.pt", map_location=device, weights_only=False))
model_lfista.eval()

# importation des résultats d'ISTA et FISTA (train_ista.py / train_fista.py, 1000 itérations sur le même test)
nmse_history_ista = np.load(resultats_dir / "resultats_ista.npz")["nmse_history_ista"]
nmse_history_fista = np.load(resultats_dir / "resultats_fista.npz")["nmse_history_fista"]


# ---------------------------------------------------------------
# 1) Nombre de paramètres appris
# ---------------------------------------------------------------
n_param_lista = sum(p.numel() for p in model_lista.parameters())
n_param_lfista = sum(p.numel() for p in model_lfista.parameters())
print(f"\nParamètres appris : ISTA 0 | FISTA 0 | LISTA {n_param_lista:,} ({n_param_lista * 4 / 1e6:.1f} Mo) "
      f"| LFISTA {n_param_lfista:,} ({n_param_lfista * 4 / 1e6:.1f} Mo)")


# ---------------------------------------------------------------
# 2) Précision atteinte et nombre d'itérations nécessaires à ISTA / FISTA pour faire aussi bien
# ---------------------------------------------------------------
with torch.no_grad():
    nmse_lista = model_lista.compute_nmse_db(model_lista(torch.from_numpy(b_test)), torch.from_numpy(x_true))
    nmse_lfista = model_lfista.compute_nmse_db(model_lfista(torch.from_numpy(b_test)), torch.from_numpy(x_true))

def iterations_pour_atteindre(nmse_history, objectif):
    # Première itération où la NMSE passe sous l'objectif (None si jamais atteint)
    for k, nmse in enumerate(nmse_history):
        if nmse <= objectif:
            return k
    return None

n_ista_lista = iterations_pour_atteindre(nmse_history_ista, nmse_lista)
n_fista_lista = iterations_pour_atteindre(nmse_history_fista, nmse_lista)
n_ista_lfista = iterations_pour_atteindre(nmse_history_ista, nmse_lfista)
n_fista_lfista = iterations_pour_atteindre(nmse_history_fista, nmse_lfista)

def texte_iter(n):
    return f"{n}" if n is not None else f"> {len(nmse_history_ista) - 1}"

print(f"\nNMSE : LISTA {nmse_lista:.2f} dB | LFISTA {nmse_lfista:.2f} dB")
print(f"Itérations pour atteindre LISTA  : ISTA {texte_iter(n_ista_lista)} | FISTA {texte_iter(n_fista_lista)}")
print(f"Itérations pour atteindre LFISTA : ISTA {texte_iter(n_ista_lfista)} | FISTA {texte_iter(n_fista_lfista)}")


# ---------------------------------------------------------------
# 3) Temps d'inférence (sans calcul de la NMSE à chaque itération)
# ---------------------------------------------------------------
def chrono(fonction):
    # Médiane de n_repet exécutions, après une exécution "à blanc" (la première est souvent plus lente)
    fonction()
    temps = []
    for _ in range(n_repet):
        t0 = time.perf_counter()
        fonction()
        temps.append(time.perf_counter() - t0)
    return float(np.median(temps))

def run_ista(b, n_iter):
    # On appelle directement step_ista : forward() calcule la NMSE à chaque itération, ce qui fausserait le temps
    model = ISTA(A=A, b=b, L=L, lag=lam)
    X = np.zeros((b.shape[0], A.shape[1]), dtype=b.dtype)
    for _ in range(n_iter):
        X = model.step_ista(X)
    return X

def run_fista(b, n_iter):
    model = FISTA(A=A, b=b, L=L, lag=lam)
    X = np.zeros((b.shape[0], A.shape[1]), dtype=b.dtype)
    v = X.copy()
    t = 1.0
    for _ in range(n_iter):
        X, v, t = model.step_fista(X, v, t)
        # step_fista renvoie v en float64 (t_new est un np.float64) : on le remet en float32,
        # sinon chaque produit avec A (float32) mélange les types et FISTA est artificiellement ralenti
        v = v.astype(b.dtype, copy=False)
    return X

def run_reseau(model, b):
    with torch.no_grad():
        return model(torch.from_numpy(b))

# Deux cas : tout le test d'un coup (on divise par n_test), et un seul problème
cas = {"batch": (b_test, n_test), "unitaire": (b_test[:1], 1)}

temps = {}   # temps par problème, en secondes
for nom_cas, (b, nb) in cas.items():
    print(f"\nChronométrage ({nom_cas}, {nb} problème(s) à la fois)...")
    temps[nom_cas] = {
        f"ISTA ({K} it.)": chrono(lambda: run_ista(b, K)) / nb,
        f"ISTA ({1000} it.)": chrono(lambda: run_ista(b, 1000)) / nb,
        f"FISTA ({K} it.)": chrono(lambda: run_fista(b, K)) / nb,
        f"LISTA ({K} couches)": chrono(lambda: run_reseau(model_lista, b)) / nb,
        f"LFISTA ({K} couches)": chrono(lambda: run_reseau(model_lfista, b)) / nb,
    }
    # FISTA jusqu'à la précision de LISTA / LFISTA (comparaison à précision égale)
    if n_fista_lista is not None:
        temps[nom_cas][f"FISTA ({n_fista_lista} it., = LISTA)"] = chrono(lambda: run_fista(b, n_fista_lista)) / nb
    if n_fista_lfista is not None:
        temps[nom_cas][f"FISTA ({n_fista_lfista} it., = LFISTA)"] = chrono(lambda: run_fista(b, n_fista_lfista)) / nb

    for methode, t in temps[nom_cas].items():
        print(f"  {methode:30s} : {t * 1e6:10.1f} µs / problème")


# ---------------------------------------------------------------
# Sauvegarde des résultats
# ---------------------------------------------------------------
methodes_batch = list(temps["batch"].keys())
np.savez(resultats_dir / "resultats_cout.npz",
         n_param_lista=n_param_lista, n_param_lfista=n_param_lfista,
         nmse_lista=nmse_lista, nmse_lfista=nmse_lfista,
         n_fista_lista=-1 if n_fista_lista is None else n_fista_lista,
         n_fista_lfista=-1 if n_fista_lfista is None else n_fista_lfista,
         methodes=methodes_batch,
         temps_batch=[temps["batch"][m] for m in methodes_batch],
         temps_unitaire=[temps["unitaire"][m] for m in methodes_batch])
print(f"\nRésultats sauvegardés dans {resultats_dir / 'resultats_cout.npz'}")


# ---------------------------------------------------------------
# Affichage graphique : temps par problème pour chaque méthode
# ---------------------------------------------------------------
couleurs = {"ISTA": "navy", "FISTA": "red", "LISTA": "darkorange", "LFISTA": "chartreuse"}

plt.figure(figsize=(14, 5))
for i, nom_cas in enumerate(cas):
    plt.subplot(1, 2, i + 1)
    noms = list(temps[nom_cas].keys())
    valeurs = [temps[nom_cas][m] * 1e6 for m in noms]
    plt.barh(noms, valeurs, color=[couleurs[m.split(" ")[0]] for m in noms])
    plt.xscale("log")
    plt.xlabel("Temps par problème (µs, échelle log)")
    plt.title("Tout le test d'un coup (1024 problèmes)" if nom_cas == "batch" else "Un seul problème à la fois")
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.gca().invert_yaxis()

plt.suptitle(f"Coût d'inférence (noiseless {A.shape[0]}x{A.shape[1]})")
plt.tight_layout()
plt.savefig(resultats_dir / "cout.png", dpi=150)
plt.show()
