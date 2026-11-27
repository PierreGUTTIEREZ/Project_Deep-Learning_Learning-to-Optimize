import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from project_ISTA_VS_LISTA.models.ista import ISTA
from project_ISTA_VS_LISTA.models.fista import FISTA
from project_ISTA_VS_LISTA.models.lista import LISTA
from project_ISTA_VS_LISTA.models.lista_cp import LISTA_CP
from project_ISTA_VS_LISTA.models.lfista import LFISTA

data_dir = BASE_DIR / "data" / "noiseless_256_512"
resultats_dir = BASE_DIR / "project_ISTA_VS_LISTA" / "resultats_train"

# Gestion automatique GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Périphérique sélectionné : {device} | Threads PyTorch : {torch.get_num_threads()}")

# ---------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------
lam = 0.005    # doit être le même que dans les scripts d'entraînement
K = 16         # nombre de couches des réseaux

# Objectifs de convergence (NMSE en dB) : on compare le temps nécessaire pour les atteindre
# -10 dB : objectif principal / -15 dB : objectif exigeant (atteint de justesse par ISTA, LISTA et LFISTA)
objectifs = [-10.0]

n_repet = 10   # chaque temps est mesuré n_repet fois, on garde la médiane

# ---------------------------------------------------------------
# Données et modèles
# ---------------------------------------------------------------
A = np.load(data_dir / "A.npy")
data = np.load(data_dir / "test_data.npy")

M = A.shape[0]  # 256
b_test = data[:, :M]       # Shape: (1024, 256)
x_true = data[:, M:]       # Shape: (1024, 512)
n_test = b_test.shape[0]

L = np.linalg.svd(A, compute_uv=False)[0]**2
print(f"Dimensions A: {A.shape}, L = {L:.4f}")

def charger_reseau(classe, nom_fichier):
    model = classe(A=A, L=L, lag=lam, K=K).to(device)
    model.load_state_dict(torch.load(resultats_dir / nom_fichier, map_location=device, weights_only=False))
    model.eval()
    return model

reseaux = {
    "LISTA": charger_reseau(LISTA, f"lista_K{K}.pt"),
    "LISTA-CP": charger_reseau(LISTA_CP, f"lista_cp_K{K}.pt"),
    "LFISTA": charger_reseau(LFISTA, f"lfista_K{K}.pt"),
}
methodes = ["ISTA", "FISTA"] + list(reseaux.keys())

# Trajectoires de NMSE sur le test de référence
# ISTA / FISTA : résultats de train_ista.py / train_fista.py (1000 itérations)
trajectoires = {
    "ISTA": np.load(resultats_dir / "resultats_ista.npz")["nmse_history_ista"],
    "FISTA": np.load(resultats_dir / "resultats_fista.npz")["nmse_history_fista"],
}
b_t = torch.from_numpy(b_test).float().to(device)
x_t = torch.from_numpy(x_true).float().to(device)
with torch.no_grad():
    for nom, model in reseaux.items():
        x_layers = model(b_t, return_all=True)
        trajectoires[nom] = [0.0] + [model.compute_nmse_db(xk, x_t) for xk in x_layers]  # 0 dB pour x^(0) = 0


# ---------------------------------------------------------------
# 1) Nombre de paramètres appris
# ---------------------------------------------------------------
n_param = {"ISTA": 0, "FISTA": 0}
for nom, model in reseaux.items():
    n_param[nom] = sum(p.numel() for p in model.parameters())

print("\nParamètres appris :")
for m in methodes:
    print(f"  {m:9s} : {n_param[m]:>12,} ({n_param[m] * 4 / 1e6:.1f} Mo)")


# ---------------------------------------------------------------
# 2) Itérations / couches nécessaires pour atteindre chaque objectif
# ---------------------------------------------------------------
def iterations_pour_atteindre(trajectoire, objectif):
    # Première itération / couche où la NMSE passe sous l'objectif (None si jamais atteint)
    for k, nmse in enumerate(trajectoire):
        if nmse <= objectif:
            return k
    return None

n_iter_objectif = {obj: {m: iterations_pour_atteindre(trajectoires[m], obj) for m in methodes} for obj in objectifs}

print("\nItérations / couches nécessaires (max : ISTA/FISTA 1000, réseaux 16)")
print(f"  {'':>8s}" + "".join(f"{m:>12s}" for m in methodes))
for obj in objectifs:
    ligne = "".join(f"{('non atteint' if n is None else str(n)):>12s}" for n in n_iter_objectif[obj].values())
    print(f"  {obj:>5.0f} dB" + ligne)


# ---------------------------------------------------------------
# 3) Temps pour atteindre chaque objectif (sans calcul de la NMSE à chaque itération)
# ---------------------------------------------------------------
def synchro():
    # Sur GPU, les calculs sont asynchrones : il faut attendre leur fin avant d'arrêter le chrono
    if device.type == "cuda":
        torch.cuda.synchronize()

def chrono(fonction):
    # Médiane de n_repet exécutions, après une exécution "à blanc" (la première est souvent plus lente)
    fonction()
    synchro()
    temps = []
    for _ in range(n_repet):
        t0 = time.perf_counter()
        fonction()
        synchro()
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

def run_reseau(model, b, n_couches):
    # On n'exécute que les n_couches premières couches : forward() boucle sur range(self.K),
    # on réduit donc K le temps de l'appel puis on remet la valeur d'origine
    K_complet = model.K
    model.K = n_couches
    try:
        with torch.no_grad():
            return model(b)
    finally:
        model.K = K_complet

def run(methode, b_np, b_torch, n):
    if methode == "ISTA":
        return run_ista(b_np, n)
    if methode == "FISTA":
        return run_fista(b_np, n)
    return run_reseau(reseaux[methode], b_torch, n)

# Deux cas : tout le test d'un coup (on divise par n_test), et un seul problème
cas = {
    "batch": (b_test, b_t, n_test),
    "unitaire": (b_test[:1], b_t[:1], 1),
}

temps = {}   # temps[cas][objectif][méthode] = temps par problème (s), nan si objectif non atteint
for nom_cas, (b_np, b_torch, nb) in cas.items():
    print(f"\nTemps pour atteindre l'objectif ({nom_cas}, {nb} problème(s) à la fois), en µs / problème")
    print(f"  {'':>8s}" + "".join(f"{m:>12s}" for m in methodes))
    temps[nom_cas] = {}
    for obj in objectifs:
        temps[nom_cas][obj] = {}
        for m in methodes:
            n = n_iter_objectif[obj][m]
            temps[nom_cas][obj][m] = np.nan if n is None else chrono(lambda: run(m, b_np, b_torch, n)) / nb
        ligne = "".join(f"{('—' if np.isnan(t) else f'{t * 1e6:.1f}'):>12s}" for t in temps[nom_cas][obj].values())
        print(f"  {obj:>5.0f} dB" + ligne)

# Gain de chaque réseau par rapport à FISTA, à précision égale
print("\nAccélération par rapport à FISTA à précision égale (temps FISTA / temps du réseau)")
for nom_cas in cas:
    for obj in objectifs:
        t_fista = temps[nom_cas][obj]["FISTA"]
        gains = []
        for nom in reseaux:
            t_res = temps[nom_cas][obj][nom]
            gains.append(f"{nom} x{t_fista / t_res:.1f}" if not np.isnan(t_res) else f"{nom} —")
        print(f"  {nom_cas:9s} {obj:>5.0f} dB : " + " | ".join(gains))


# ---------------------------------------------------------------
# Sauvegarde des résultats (nan = objectif non atteint)
# ---------------------------------------------------------------
np.savez(resultats_dir / "resultats_cout.npz",
         methodes=methodes, objectifs=objectifs,
         n_param=[n_param[m] for m in methodes],
         n_iter_objectif=[[np.nan if n_iter_objectif[o][m] is None else n_iter_objectif[o][m] for m in methodes] for o in objectifs],
         temps_batch=[[temps["batch"][o][m] for m in methodes] for o in objectifs],
         temps_unitaire=[[temps["unitaire"][o][m] for m in methodes] for o in objectifs])
print(f"\nRésultats sauvegardés dans {resultats_dir / 'resultats_cout.npz'}")


# ---------------------------------------------------------------
# Affichage graphique : temps pour atteindre chaque objectif
# ---------------------------------------------------------------
couleurs = {"ISTA": "navy", "FISTA": "red", "LISTA": "darkorange", "LISTA-CP": "magenta", "LFISTA": "chartreuse"}
x_pos = np.arange(len(objectifs))
largeur = 0.16

plt.figure(figsize=(14, 5.5))
for i, nom_cas in enumerate(cas):
    plt.subplot(1, 2, i + 1)
    for j, m in enumerate(methodes):
        valeurs = [temps[nom_cas][o][m] * 1e6 for o in objectifs]
        decalage = (j - len(methodes) / 2 + 0.5) * largeur
        barres = plt.bar(x_pos + decalage, np.nan_to_num(valeurs), width=largeur, color=couleurs[m], label=m)
        # nombre d'itérations / couches au-dessus de chaque barre
        for barre, o, v in zip(barres, objectifs, valeurs):
            n = n_iter_objectif[o][m]
            texte = "non\natteint" if n is None else f"{n} it."
            hauteur = barre.get_height() if not np.isnan(v) else plt.ylim()[0]
            plt.annotate(texte, xy=(barre.get_x() + barre.get_width() / 2, max(hauteur, 1e-9)),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    plt.yscale("log")
    plt.xticks(x_pos, [f"{o:.0f} dB" for o in objectifs])
    plt.xlabel("Objectif de NMSE")
    plt.ylabel("Temps par problème (µs, échelle log)")
    plt.title("Tout le test d'un coup (1024 problèmes)" if nom_cas == "batch" else "Un seul problème à la fois")
    plt.grid(True, axis="y", which="both", linestyle="--", alpha=0.5)
    if i == 0:
        plt.legend(fontsize=8)

plt.suptitle(f"Coût pour atteindre un objectif de NMSE (noiseless {A.shape[0]}x{A.shape[1]}, {device})")
plt.tight_layout()
plt.savefig(resultats_dir / "cout.png", dpi=150)
plt.show()
