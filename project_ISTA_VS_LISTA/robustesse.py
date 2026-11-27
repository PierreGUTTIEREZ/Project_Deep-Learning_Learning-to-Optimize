import sys
import csv
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
robu_dir = BASE_DIR / "data" / "robustesse"
resultats_dir = BASE_DIR / "project_ISTA_VS_LISTA" / "resultats_train"

# Gestion automatique GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Périphérique sélectionné : {device}")

rng = np.random.default_rng(1)   # pour que les perturbations générées ici soient reproductibles

# Données de référence (celles de l'entraînement)
A = np.load(data_dir / "A.npy")
data_ref = np.load(data_dir / "test_data.npy")

M = A.shape[0]  # 256
b_ref = data_ref[:, :M]       # Shape: (1024, 256)
x_ref = data_ref[:, M:]       # Shape: (1024, 512)

L = np.linalg.svd(A, compute_uv=False)[0]**2
print(f"Dimensions A: {A.shape}, L = {L:.4f}")

lam = 0.005    # doit être le même que dans les scripts d'entraînement
K = 16         # nombre de couches des réseaux entraînés

# ---------------------------------------------------------------
# Modèles calibrés : chaque méthode est arrêtée au nombre d'itérations / couches
# qu'il lui faut pour atteindre -10 dB sur les données de référence (paliers_benchmark.py).
# Elles sont donc toutes à la même précision au départ, et on regarde comment elles se dégradent.
# ---------------------------------------------------------------
palier = 10   # en dB (colonne "Palier_10dB" du benchmark)

noms_csv = {"ISTA": "ISTA", "FISTA": "FISTA", f"LISTA (K={K})": "LISTA",
            f"LISTA-CP (K={K})": "LISTA-CP", f"LFISTA (K={K})": "LFISTA"}
n_iter = {}
with open(resultats_dir / "paliers_iterations.csv", encoding="utf-8") as f:
    for ligne in csv.DictReader(f):
        if ligne["Algorithme"] in noms_csv:
            n_iter[noms_csv[ligne["Algorithme"]]] = int(ligne[f"Palier_{palier}dB"])

methodes = ["ISTA", "FISTA", "LISTA", "LISTA-CP", "LFISTA"]
print("Nombre d'itérations / couches pour -10 dB (benchmark) : "
      + " | ".join(f"{m} {n_iter[m]}" for m in methodes))

# importation des réseaux (entraînés sur les données de référence)
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
    # NMSE (dB) de chaque modèle calibré sur un scénario
    # ISTA / FISTA utilisent la "vraie" matrice du scénario, les réseaux gardent les poids appris avec A
    L_utilisee = np.linalg.svd(A_utilisee, compute_uv=False)[0]**2
    X_init = np.zeros_like(x_true)
    nmse = {}

    model_ista = ISTA(A=A_utilisee, b=b, L=L_utilisee, lag=lam, x_true=x_true)
    _, nmse_history_ista = model_ista.forward(X_init, epochs=n_iter["ISTA"])
    nmse["ISTA"] = nmse_history_ista[-1]

    model_fista = FISTA(A=A_utilisee, b=b, L=L_utilisee, lag=lam, x_true=x_true)
    _, nmse_history_fista = model_fista.forward(X_init, epochs=n_iter["FISTA"])
    nmse["FISTA"] = nmse_history_fista[-1]

    b_t = torch.from_numpy(b).float().to(device)
    x_t = torch.from_numpy(x_true).float().to(device)
    with torch.no_grad():
        for nom, model in reseaux.items():
            x_layers = model(b_t, return_all=True)
            nmse[nom] = model.compute_nmse_db(x_layers[n_iter[nom] - 1], x_t)   # sortie de la couche n_iter[nom]

    return [nmse[m] for m in methodes]


# ---------------------------------------------------------------
# Scénarios : (titre, valeurs testées, fonction qui renvoie (b, x*, A du scénario))
# ---------------------------------------------------------------
def scenario_p(p):
    return (b_ref, x_ref, A) if p == 0.1 else (*charger_scenario(f"p{p:.2f}"), A)

def scenario_snr(snr):
    if snr == np.inf:
        return b_ref, x_ref, A
    if snr == 20:
        return (*charger_scenario("snr20"), A)
    return ajouter_bruit(b_ref, snr), x_ref, A

def scenario_eps(eps):
    if eps == 0.0:
        return b_ref, x_ref, A
    A_pert = perturber_A(eps)
    return (x_ref @ A_pert.T).astype(np.float32), x_ref, A_pert

def scenario_amp(amp):
    x = (amp * x_ref).astype(np.float32)
    return (x @ A.T).astype(np.float32), x, A

groupes = {
    "p": ("Parcimonie p (entraînement : 0.1)", [0.05, 0.1, 0.15, 0.2, 0.3], scenario_p),
    "snr": ("SNR en dB (entraînement : sans bruit)", [np.inf, 40, 30, 20, 10], scenario_snr),
    "eps": ("Perturbation ε de A (entraînement : 0)", [0.0, 0.01, 0.05, 0.1], scenario_eps),
    "amp": ("Amplitude de x* (entraînement : 1)", [0.5, 1.0, 2.0, 5.0], scenario_amp),
}

resultats = {}   # resultats[groupe] = NMSE (n_valeurs, n_methodes)
for cle, (titre, valeurs, scenario) in groupes.items():
    print(f"{titre}...")
    resultats[cle] = np.array([evaluer(*scenario(v)) for v in valeurs])

# Sauvegarde des résultats (une ligne par valeur testée, une colonne par méthode)
sauvegarde = {"methodes": methodes, "n_iter": [n_iter[m] for m in methodes]}
for cle, (titre, valeurs, _) in groupes.items():
    sauvegarde[f"{cle}_values"] = valeurs
    sauvegarde[f"nmse_{cle}"] = resultats[cle]
np.savez(resultats_dir / "resultats_robustesse.npz", **sauvegarde)
print(f"Résultats sauvegardés dans {resultats_dir / 'resultats_robustesse.npz'}")


# ---------------------------------------------------------------
# Affichage dans le terminal : NMSE (dB) de chaque modèle calibré
# ---------------------------------------------------------------
entete = "".join(f"{f'{m} ({n_iter[m]})':>16s}" for m in methodes)
for cle, (titre, valeurs, _) in groupes.items():
    print(f"\n{titre} — NMSE en dB")
    print(f"{'':>8s}" + entete)
    for v, ligne in zip(valeurs, resultats[cle]):
        print(f"{str(v):>8s}" + "".join(f"{n:>16.2f}" for n in ligne))


# ---------------------------------------------------------------
# Affichage graphique
# ---------------------------------------------------------------
couleurs = {"ISTA": "navy", "FISTA": "red", "LISTA": "darkorange", "LISTA-CP": "magenta", "LFISTA": "chartreuse"}
marqueurs = {"ISTA": "o", "FISTA": "o", "LISTA": "s", "LISTA-CP": "D", "LFISTA": "^"}

plt.figure(figsize=(14, 9))
for i, (cle, (titre, valeurs, _)) in enumerate(groupes.items()):
    plt.subplot(2, 2, i + 1)
    labels = ["∞" if v == np.inf else str(v) for v in valeurs]
    for j, m in enumerate(methodes):
        plt.plot(labels, resultats[cle][:, j], marker=marqueurs[m], color=couleurs[m],
                 label=f"{m} ({n_iter[m]} it.)", lw=2)
    plt.axhline(-palier, color="gray", linestyle=":", lw=1.5)   # niveau de départ commun
    plt.xlabel(titre)
    plt.ylabel("NMSE (dB)")
    plt.grid(True, linestyle="--", alpha=0.6)
    if i == 0:
        plt.legend(fontsize=8)

plt.suptitle(f"Robustesse à précision égale : chaque méthode est calibrée pour atteindre -{palier} dB "
             f"sur les données d'entraînement")
plt.tight_layout()
plt.savefig(resultats_dir / "robustesse.png", dpi=150)
plt.show()
