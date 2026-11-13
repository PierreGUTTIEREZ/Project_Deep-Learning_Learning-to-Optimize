"""
Génération des données pour l'exemple 1 (Convex Sparse Optimization / LASSO)
du benchmark "Learning to Optimize: A Primer and A Benchmark" (Chen et al., 2021),
en suivant le tirage i.i.d. de Chen, Liu, Wang & Yin (NeurIPS 2018).

Le problème : on observe  b_q = A x*_q + e_q  et on veut retrouver x*_q (parcimonieux).
A est FIXE pour tous les échantillons (train, val et test), seuls x*_q et e_q changent.

Tirage (section "Data generation" du Primer) :
    - x*_q : composantes i.i.d. selon Ber(0.1) * N(0, 1)  (~10 % de coordonnées non nulles)
    - A    : A_ij ~ N(0, 1/m), puis colonnes normalisées (norme 2 = 1)
    - 51 200 échantillons d'entraînement, 1 024 de validation, 1 024 de test

Les 4 réglages de l'article (option --setting) :
    - noiseless : (m, n) = (256, 512), pas de bruit
    - noisy     : idem, avec un bruit gaussien de SNR = 20 dB
    - coherent  : idem que noiseless, mais A = dictionnaire D (256 x 512) appris sur
                  400 images de BSD500. Ce dictionnaire n'est pas généré ici : il faut
                  le fournir avec --dict chemin/vers/D.npy
    - large     : (m, n) = (512, 1024), pas de bruit

Format attendu par Model_Base_L2O/train.py (cf. data_preprocessing.py) :
    - A.npy                         : la matrice, forme (m, n)
    - train_data.npy, val_data.npy,
      test_data.npy                 : une ligne par échantillon = [b, x*], forme (nb, m + n)

Exemples :
    python data/gen_data.py --setting noiseless
    python data/gen_data.py --setting noisy
    python data/gen_data.py --setting coherent --dict data/D_256_512.npy
    python data/gen_data.py --setting large
"""

import argparse
import os

import numpy as np


# Paramètres de chaque réglage : (m, n, SNR en dB ou None si pas de bruit)
SETTINGS = {
    "noiseless": (256, 512, None),
    "noisy":     (256, 512, 20),
    "coherent":  (256, 512, None),
    "large":     (512, 1024, None),
}


def gen_A(m, n, rng):
    """Matrice de mesure gaussienne avec des colonnes de norme 1."""
    A = rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, n))
    return normalize_columns(A)


def normalize_columns(A):
    """Ramène chaque colonne de A à une norme 2 égale à 1."""
    return A / np.linalg.norm(A, axis=0, keepdims=True)


def gen_x_star(nb, n, p, rng):
    """nb vecteurs parcimonieux de taille n (un par ligne), selon Ber(p) * N(0, 1)."""
    support = rng.binomial(1, p, size=(nb, n))  # 1 = coordonnée active
    valeurs = rng.normal(0.0, 1.0, size=(nb, n))
    return support * valeurs


def gen_b(A, x_star, snr, rng):
    """Mesures b = A x* + e (un échantillon par ligne).

    snr : rapport signal sur bruit en dB. None => pas de bruit.
    """
    b = x_star @ A.T  # équivaut à A @ x pour chaque ligne x de x_star

    if snr is not None:
        # On règle la variance du bruit pour que ||A x||^2 / ||e||^2 = 10^(snr/10)
        puissance_signal = np.mean(b ** 2, axis=1, keepdims=True)
        sigma = np.sqrt(puissance_signal / 10 ** (snr / 10))
        b = b + sigma * rng.normal(size=b.shape)

    return b


def gen_dataset(A, nb, p, snr, rng):
    """Construit un jeu de nb échantillons au format [b, x*]."""
    n = A.shape[1]
    x_star = gen_x_star(nb, n, p, rng)
    b = gen_b(A, x_star, snr, rng)
    return np.concatenate([b, x_star], axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Génère les données LASSO (exemple 1 du Primer L2O).")
    parser.add_argument("--setting", choices=SETTINGS.keys(), default="noiseless",
                        help="un des 4 réglages de l'article")
    parser.add_argument("--dict", type=str, default=None,
                        help="fichier .npy du dictionnaire D (obligatoire pour 'coherent')")
    parser.add_argument("--p", type=float, default=0.1, help="proba qu'une coordonnée soit non nulle")
    parser.add_argument("--n_train", type=int, default=51200)
    parser.add_argument("--n_val", type=int, default=1024)
    parser.add_argument("--n_test", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None,
                        help="dossier de sortie (par défaut : data/<setting>_<m>_<n> à côté de ce script)")
    args = parser.parse_args()

    m, n, snr = SETTINGS[args.setting]
    rng = np.random.default_rng(args.seed)

    # 1) La matrice A, commune à tous les échantillons
    if args.setting == "coherent":
        if args.dict is None:
            parser.error("le réglage 'coherent' demande le dictionnaire D : --dict chemin/vers/D.npy")
        A = normalize_columns(np.load(args.dict))
        if A.shape != (m, n):
            parser.error(f"le dictionnaire doit être de taille {(m, n)}, reçu {A.shape}")
    else:
        A = gen_A(m, n, rng)
    A = A.astype(np.float32)

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"{args.setting}_{m}_{n}")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "A.npy"), A)

    # 2) Les trois jeux de données (mêmes A, nouveaux x* et bruits)
    tailles = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for nom, nb in tailles.items():
        data = gen_dataset(A, nb, args.p, snr, rng)
        np.save(os.path.join(out_dir, f"{nom}_data.npy"), data)
        print(f"{nom:5s} : {data.shape}  ->  {nom}_data.npy")

    bruit = "sans bruit" if snr is None else f"SNR = {snr} dB"
    print(f"Réglage '{args.setting}' : A {A.shape}, p = {args.p}, {bruit}")
    print(f"Fichiers enregistrés dans {out_dir}")


if __name__ == "__main__":
    main()
