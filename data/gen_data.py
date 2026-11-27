import argparse
import os
import numpy as np

SETTINGS = {
    "noiseless": (256, 512, None),
    "noisy":     (256, 512, 20),
    "coherent":  (256, 512, None),
    "large":     (512, 1024, None),
}

def gen_A(m, n, rng):
    A = rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, n))
    return normalize_columns(A)

def normalize_columns(A):
    return A / np.linalg.norm(A, axis=0, keepdims=True)

def gen_x_star(nb, n, p, rng):
    support = rng.binomial(1, p, size=(nb, n))
    valeurs = rng.normal(0.0, 1.0, size=(nb, n))
    return support * valeurs

def gen_b(A, x_star, snr, rng):
    b = x_star @ A.T

    if snr is not None:
        puissance_signal = np.mean(b ** 2, axis=1, keepdims=True)
        sigma = np.sqrt(puissance_signal / 10 ** (snr / 10))
        b = b + sigma * rng.normal(size=b.shape)

    return b

def gen_dataset(A, nb, p, snr, rng):
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
