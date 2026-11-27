import numpy as np
import torch
import torch.nn as nn


class LISTA_CP(nn.Module):
    """
    LISTA-CP : Learned ISTA with Coupled Parameters (Chen et al., NeurIPS 2018).
    
    Contrairement à LISTA standard où W1 et W2 sont deux matrices libres,
    LISTA-CP impose la contrainte structurelle issue d'ISTA :
        W2^(k) = I - W1^(k) A
        
    Mise à jour à la couche k :
        res^(k) = b - x^(k) A^T
        z^(k)   = x^(k) + W_k(res^(k))
        x^(k+1) = S_{theta_k}(z^(k))
        
    Avantages théoriques et pratiques :
      1. Nombre de paramètres divisé par 3 (pas de matrice N x N à apprendre).
      2. Preuve de convergence linéaire rigoureuse.
      3. Moins de surapprentissage et meilleure stabilité numérique.
    """

    def __init__(self, A: np.ndarray, L: float, lag: float, K: int):
        super(LISTA_CP, self).__init__()
        self.L = L
        self.lag = lag
        self.K = K
        M, N = A.shape
        self.M = M
        self.N = N

        # Matrice A enregistrée comme buffer constant (sur le bon device)
        self.register_buffer("A", torch.from_numpy(A).float())

        # Seules les matrices W_k (taille M -> N) et les seuils theta_k sont appris
        self.W = nn.ModuleList([nn.Linear(self.M, self.N, bias=False) for _ in range(K)])
        self.theta = nn.ParameterList([nn.Parameter(torch.tensor(lag / L)) for _ in range(K)])

        # Initialisation avec les poids théoriques d'ISTA : W = A^T / L
        A_t = torch.from_numpy(A).float()
        W_ista = A_t.T / L
        for k in range(K):
            self.W[k].weight.data = W_ista.clone()

    def soft_threshold(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.sign(z) * torch.clamp(torch.abs(z) - theta, min=0.0)

    def forward(self, b: torch.Tensor, return_all: bool = False):
        x = torch.zeros(b.shape[0], self.N, device=b.device)
        x_history = []

        for k in range(self.K):
            # Calcul du résidu : res = b - x @ A^T (taille B x M)
            residual = b - x @ self.A.T
            # Mise à jour couplée : z = x + res @ W_k^T (taille B x N)
            z = x + self.W[k](residual)
            # Seuillage doux
            x = self.soft_threshold(z, self.theta[k])
            x_history.append(x)

        if return_all:
            return x_history
        return x

    def compute_nmse_db(self, X: torch.Tensor, x_true: torch.Tensor) -> float:
        mse = torch.mean(torch.norm(X - x_true, dim=1)**2)
        norm_true = torch.mean(torch.norm(x_true, dim=1)**2)
        return 10.0 * torch.log10(mse / norm_true).item()
