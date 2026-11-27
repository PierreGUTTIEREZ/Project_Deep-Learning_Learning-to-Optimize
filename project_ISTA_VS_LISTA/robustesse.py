import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))   # pour importer les classes (dans ce dossier)
from ista import ISTA
from fista import FISTA
from lista import LISTA
from lfista import LFISTA

data_dir = BASE_DIR / "data" / "noiseless_256_512"
robu_dir = BASE_DIR / "data" / "robustesse"
resultats_dir = BASE_DIR / "project_ISTA_VS_LISTA" / "resultats_train"

rng = np.random.default_rng(1)   # pour que les perturbations générées ici soient reproductibles

# Données de référence (celles de l'entraînement)
A = np.load(data_dir / "A.npy")
data_ref = np.load(data_dir / "test_data.npy")

M = A.shape[0]  # 256
b_ref = data_ref[:, :M]       # Shape: (1024, 256)
x_ref = data_ref[:, M:]       # Shape: (1024, 512)

L = np.linalg.svd(A, compute_uv=False)[0]**2
print(f"Dimensions A: {A.shape}, L = {L:.4f}")

lam = 0.005    # doit être le même que dans train_lista.py / train_lfista.py
K = 16         # nombre de couches de LISTA / LFISTA (et nombre d'itérations pour la comparaison "à égalité")
n_iter = 1000  # nombre d'itérations pour la "meilleure" solution classique

# importation de LISTA et LFISTA (entraînés sur les données de référence)
model_lista = LISTA(A=A, L=L, lag=lam, K=K)
model_lista.load_state_dict(torch.load(resultats_dir / f"lista_K{K}.pt"))
model_lista.eval()

model_lfista = LFISTA(A=A, L=L, lag=lam, K=K)
model_lfista.load_state_dict(torch.load(resultats_dir / f"lfista_K{K}.pt"))
model_lfista.eval()


# ---------------------------------------------------------------
# Fonctions utiles
# ---------------------------------------------------------------
def charger_scenario(nom_dossier):
    # Charge un jeu de test généré par gen_data.py et vérifie que A est bien la même
    A_robu = np.load(robu_dir / nom_dossier / "A.npy")
    if not np.allclose(A_robu, A):
        sys.exit(f"La matrice A de {nom_dossier} est différente de celle de l'entraînement : comparaison impossible.")
    data = np.load(robu_dir / nom_dossier / "test_data.npy")
    return data[:, :M], data[:, M:]

def ajouter_bruit(b, snr):
    # Même formule que gen_b dans data/gen_data.py
    puissance_signal = np.mean(b ** 2, axis=1, keepdims=True)
    sigma = np.sqrt(puissance_signal / 10 ** (snr / 10))
    return (b + sigma * rng.normal(size=b.shape)).astype(np.float32)

def perturber_A(eps):
    # A + eps * bruit gaussien (même loi que A), puis colonnes renormalisées comme dans gen_data.py
    A_pert = A + eps * rng.normal(0.0, 1.0 / np.sqrt(M), size=A.shape)
    A_pert = A_pert / np.linalg.norm(A_pert, axis=0, keepdims=True)
    return A_pert.astype(np.float32)

def evaluer(b, x_true, A_utilisee):
    # NMSE (dB) de chaque méthode sur un scénario
    # ISTA / FISTA utilisent la "vraie" matrice du scénario, LISTA / LFISTA gardent les poids appris avec A
    L_utilisee = np.linalg.svd(A_utilisee, compute_uv=False)[0]**2
    X_init = np.zeros_like(x_true)

    model_ista = ISTA(A=A_utilisee, b=b, L=L_utilisee, lag=lam, x_true=x_true)
    _, nmse_history_ista = model_ista.forward(X_init, epochs=n_iter)

    model_fista = FISTA(A=A_utilisee, b=b, L=L_utilisee, lag=lam, x_true=x_true)
    _, nmse_history_fista = model_fista.forward(X_init, epochs=n_iter)

    with torch.no_grad():
        b_t, x_t = torch.from_numpy(b), torch.from_numpy(x_true)
        nmse_lista = model_lista.compute_nmse_db(model_lista(b_t), x_t)
        nmse_lfista = model_lfista.compute_nmse_db(model_lfista(b_t), x_t)

    return [nmse_history_ista[-1], nmse_history_fista[100], nmse_lista, nmse_lfista]

methodes = [f"ISTA ({n_iter} it.)", f"FISTA (100 it.)", f"LISTA ({K} couches)", f"LFISTA ({K} couches)"]


# ---------------------------------------------------------------
# Scénarios
# ---------------------------------------------------------------
print("Référence...")
nmse_ref = evaluer(b_ref, x_ref, A)

# 1) Parcimonie p (entraînement : p = 0.1)
p_values = [0.05, 0.1, 0.15, 0.2, 0.3]
nmse_p = []
for p in p_values:
    print(f"Parcimonie p = {p}...")
    if p == 0.1:
        nmse_p.append(nmse_ref)
    else:
        b, x = charger_scenario(f"p{p:.2f}")
        nmse_p.append(evaluer(b, x, A))

# 2) Bruit (entraînement : sans bruit). 20 dB vient de gen_data.py, les autres sont ajoutés ici
snr_values = [np.inf, 40, 30, 20, 10]
nmse_snr = []
for snr in snr_values:
    print(f"SNR = {snr} dB...")
    if snr == np.inf:
        nmse_snr.append(nmse_ref)
    elif snr == 20:
        b, x = charger_scenario("snr20")
        nmse_snr.append(evaluer(b, x, A))
    else:
        nmse_snr.append(evaluer(ajouter_bruit(b_ref, snr), x_ref, A))

# 3) Matrice A perturbée (les réseaux ne sont pas réentraînés)
eps_values = [0.0, 0.01, 0.05, 0.1]
nmse_eps = []
for eps in eps_values:
    print(f"Perturbation de A : eps = {eps}...")
    if eps == 0.0:
        nmse_eps.append(nmse_ref)
    else:
        A_pert = perturber_A(eps)
        b = (x_ref @ A_pert.T).astype(np.float32)
        nmse_eps.append(evaluer(b, x_ref, A_pert))

# 4) Amplitude de x* (entraînement : amplitude 1)
amp_values = [0.5, 1.0, 2.0, 5.0]
nmse_amp = []
for amp in amp_values:
    print(f"Amplitude de x* : x {amp}...")
    if amp == 1.0:
        nmse_amp.append(nmse_ref)
    else:
        x = (amp * x_ref).astype(np.float32)
        b = (x @ A.T).astype(np.float32)
        nmse_amp.append(evaluer(b, x, A))

nmse_p, nmse_snr, nmse_eps, nmse_amp = np.array(nmse_p), np.array(nmse_snr), np.array(nmse_eps), np.array(nmse_amp)

# Sauvegarde des résultats (une ligne par valeur testée, une colonne par méthode)
np.savez(resultats_dir / "resultats_robustesse.npz", methodes=methodes,
         p_values=p_values, nmse_p=nmse_p,
         snr_values=snr_values, nmse_snr=nmse_snr,
         eps_values=eps_values, nmse_eps=nmse_eps,
         amp_values=amp_values, nmse_amp=nmse_amp)
print(f"Résultats sauvegardés dans {resultats_dir / 'resultats_robustesse.npz'}")


# ---------------------------------------------------------------
# Affichage dans le terminal
# ---------------------------------------------------------------
def afficher_tableau(titre, valeurs, nmse):
    print(f"\n{titre}")
    print(f"{'':>10s}" + "".join(f"{m:>20s}" for m in methodes))
    for v, ligne in zip(valeurs, nmse):
        print(f"{str(v):>10s}" + "".join(f"{n:>20.2f}" for n in ligne))

afficher_tableau("Parcimonie p (entraînement : 0.1) — NMSE en dB", p_values, nmse_p)
afficher_tableau("SNR en dB (entraînement : sans bruit = inf) — NMSE en dB", snr_values, nmse_snr)
afficher_tableau("Perturbation eps de A (entraînement : 0) — NMSE en dB", eps_values, nmse_eps)
afficher_tableau("Amplitude de x* (entraînement : 1) — NMSE en dB", amp_values, nmse_amp)


# ---------------------------------------------------------------
# Affichage graphique
# ---------------------------------------------------------------
couleurs = ["navy", "red", "darkorange", "chartreuse"]
styles = ["-", "-", "-", "-"]
marqueurs = ["o", "o", "s", "s"]

groupes = [
    ("Parcimonie p (entraînement : 0.1)", [str(v) for v in p_values], nmse_p),
    ("SNR en dB (entraînement : sans bruit)", ["∞" if v == np.inf else str(v) for v in snr_values], nmse_snr),
    ("Perturbation ε de A (entraînement : 0)", [str(v) for v in eps_values], nmse_eps),
    ("Amplitude de x* (entraînement : 1)", [str(v) for v in amp_values], nmse_amp),
]

plt.figure(figsize=(14, 9))
for i, (titre, labels, nmse) in enumerate(groupes):
    plt.subplot(2, 2, i + 1)
    for j, m in enumerate(methodes):
        plt.plot(labels, nmse[:, j], linestyle=styles[j], marker=marqueurs[j], color=couleurs[j], label=m, lw=2)
    plt.xlabel(titre)
    plt.ylabel("NMSE (dB)")
    plt.grid(True, linestyle="--", alpha=0.6)
    if i == 0:
        plt.legend(fontsize=8)

plt.suptitle(f"Robustesse hors distribution d'entraînement (noiseless {A.shape[0]}x{A.shape[1]})")
plt.tight_layout()
plt.savefig(resultats_dir / "robustesse.png", dpi=150)
plt.show()
