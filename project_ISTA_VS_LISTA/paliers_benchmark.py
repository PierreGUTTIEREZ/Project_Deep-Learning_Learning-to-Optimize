import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(CURRENT_DIR / "models"))

from project_ISTA_VS_LISTA.models.lista import LISTA
from project_ISTA_VS_LISTA.models.lfista import LFISTA
from project_ISTA_VS_LISTA.models.lista_cp import LISTA_CP

data_dir = BASE_DIR / "data" / "noiseless_256_512"
res_dir = CURRENT_DIR / "resultats_train"
res_dir.mkdir(parents=True, exist_ok=True)

# 1. Chargement des données de test
A = np.load(data_dir / "A.npy")
test_data = np.load(data_dir / "test_data.npy")
M = A.shape[0]
b_test = torch.from_numpy(test_data[:, :M])
x_true = torch.from_numpy(test_data[:, M:])
L = np.linalg.svd(A, compute_uv=False)[0]**2
lam = 0.005
K = 16

# 2. Chargement des trajectoires NMSE
trajectories = {}

# A. ISTA
data_ista = np.load(res_dir / "resultats_ista.npz")
trajectories["ISTA"] = list(data_ista["nmse_history_ista"])

# B. FISTA
data_fista = np.load(res_dir / "resultats_fista.npz")
trajectories["FISTA"] = list(data_fista["nmse_history_fista"])

# C. LISTA (K=16)
p_lista = res_dir / f"lista_K{K}.pt"
if p_lista.exists():
    m_lista = LISTA(A, L, lam, K=K)
    m_lista.load_state_dict(torch.load(p_lista, weights_only=False))
    m_lista.eval()
    with torch.no_grad():
        x_layers = m_lista(b_test, return_all=True)
    trajectories[f"LISTA (K={K})"] = [0.0] + [m_lista.compute_nmse_db(xk, x_true) for xk in x_layers]

# D. LFISTA (K=16)
p_lfista = res_dir / f"lfista_K{K}.pt"
if p_lfista.exists():
    m_lfista = LFISTA(A, L, lam, K=K)
    m_lfista.load_state_dict(torch.load(p_lfista, weights_only=False))
    m_lfista.eval()
    with torch.no_grad():
        x_layers = m_lfista(b_test, return_all=True)
    trajectories[f"LFISTA (K={K})"] = [0.0] + [m_lfista.compute_nmse_db(xk, x_true) for xk in x_layers]

# E. LISTA-CP (K=16)
p_listacp = res_dir / f"lista_cp_K{K}.pt"
if p_listacp.exists():
    m_cp = LISTA_CP(A, L, lam, K=K)
    m_cp.load_state_dict(torch.load(p_listacp, weights_only=False))
    m_cp.eval()
    with torch.no_grad():
        x_layers = m_cp(b_test, return_all=True)
    trajectories[f"LISTA-CP (K={K})"] = [0.0] + [m_cp.compute_nmse_db(xk, x_true) for xk in x_layers]

# 3. Calcul du premier indice où NMSE <= Palier
paliers = [-3.0, -5.0, -10.0, -15.0]  # en dB
table_results = []

for algo_name, traj in trajectories.items():
    row = {"Algorithme": algo_name, "NMSE_finale_dB": f"{traj[-1]:.2f}"}
    for palier in paliers:
        col_name = f"Palier_{abs(int(palier))}dB"
        reached_idx = None
        for it, val in enumerate(traj):
            if val <= palier:
                reached_idx = it
                break
        if reached_idx is not None:
            row[col_name] = reached_idx
        else:
            row[col_name] = f"> {len(traj)-1} (non atteint)"
    table_results.append(row)

# 4. Sauvegarde dans un fichier CSV
csv_path = res_dir / "paliers_iterations.csv"
fieldnames = ["Algorithme", "NMSE_finale_dB"] + [f"Palier_{abs(int(p))}dB" for p in paliers]
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in table_results:
        writer.writerow(row)
print(f"Tableau des paliers enregistré dans : {csv_path}")

# Affichage terminal au format Markdown
print("\n" + "="*70)
print("RÉSULTATS : NOMBRE D'ITÉRATIONS POUR ATTEINDRE LES PALIERS DE NMSE")
print("="*70)
header_str = f"{'Algorithme':<20} | {'NMSE finale':<12} | " + " | ".join([f"{f'{abs(int(p))} dB':<10}" for p in paliers])
print(header_str)
print("-" * len(header_str))
for row in table_results:
    vals = [str(row[f"Palier_{abs(int(p))}dB"]) for p in paliers]
    val_str = " | ".join([f"{v:<10}" for v in vals])
    print(f"{row['Algorithme']:<20} | {row['NMSE_finale_dB'] + ' dB':<12} | {val_str}")
print("="*70 + "\n")

# 5. Visualisation graphique double (Courbes avec paliers + Histogramme comparatif)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

colors = {
    "ISTA": "navy",
    "FISTA": "red",
    f"LISTA (K={K})": "darkorange",
    f"LFISTA (K={K})": "chartreuse",
    f"LISTA-CP (K={K})": "magenta",
}

# Sous-graphique 1 : Zoom sur les trajectoires et les paliers horizontaux
max_plot_it = 120
for algo_name, traj in trajectories.items():
    sub_traj = traj[:max_plot_it + 1]
    color = colors.get(algo_name, "gray")
    ls = "--" if "CP" in algo_name else "-"
    ax1.plot(range(len(sub_traj)), sub_traj, label=algo_name, color=color, lw=2, linestyle=ls)

palier_colors = ["#666666", "#444444", "#222222", "#000000"]
for p, pc in zip(paliers, palier_colors):
    ax1.axhline(p, color=pc, linestyle=":", lw=1.5, alpha=0.7)
    ax1.text(max_plot_it * 0.75, p + 0.3, f"Palier {p} dB", color=pc, fontsize=9, fontweight="bold")

ax1.set_xlabel("Itérations / Couches (zoom 0-120)", fontsize=11)
ax1.set_ylabel("NMSE (dB)", fontsize=11)
ax1.set_title("Trajectoires de convergence et Paliers cibles", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="upper right", fontsize=9)

# Sous-graphique 2 : Histogramme comparatif du nombre d'itérations pour chaque palier
algos_to_bar = [row["Algorithme"] for row in table_results]
bar_paliers = [-5.0, -10.0, -15.0]
x_pos = np.arange(len(bar_paliers))
bar_width = 0.16

for i, algo in enumerate(algos_to_bar):
    row = next(r for r in table_results if r["Algorithme"] == algo)
    y_vals = []
    for p in bar_paliers:
        raw_v = row[f"Palier_{abs(int(p))}dB"]
        if isinstance(raw_v, int):
            y_vals.append(raw_v)
        else:
            # Si non atteint, on met une barre saturée pour indiquer le dépassement
            y_vals.append(1000 if "ISTA" in algo else 25)
    offset = (i - len(algos_to_bar)/2 + 0.5) * bar_width
    color = colors.get(algo, "gray")
    bars = ax2.bar(x_pos + offset, y_vals, width=bar_width, label=algo, color=color, alpha=0.85)
    
    # Annotations sur les barres
    for bar, y in zip(bars, y_vals):
        text = str(y) if y < 900 else ">1000"
        ax2.annotate(text,
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, rotation=45 if y > 100 else 0)

ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"{abs(int(p))} dB" for p in bar_paliers], fontsize=11, fontweight="bold")
ax2.set_xlabel("Paliers de NMSE", fontsize=11)
ax2.set_ylabel("Nombre d'itérations nécessaires (échelle log)", fontsize=11)
ax2.set_yscale("log")
ax2.set_title("Comparaison du nombre d'itérations requises", fontsize=12, fontweight="bold")
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", fontsize=9)

plt.tight_layout()
fig_path = res_dir / "paliers_iterations.png"
plt.savefig(fig_path, dpi=300)
print(f"Graphique comparatif des paliers sauvegardé dans : {fig_path}")
# On évite le blocage dans les scripts non-interactifs
if hasattr(sys, 'ps1') or not sys.flags.interactive:
    plt.close()
else:
    plt.show()
