import numpy as np
import torch
import torch.nn as nn

class LISTA(nn.Module):

    def __init__(self, A: np.ndarray, L: float, lag: float, K: int):
        super(LISTA, self).__init__()
        self.L = L
        self.lag = lag
        self.K = K
        M, N = A.shape
        self.M = M
        self.N = N

        self.W1 = nn.ModuleList([nn.Linear(self.M, self.N, bias=False) for i in range(K)])
        self.W2 = nn.ModuleList([nn.Linear(self.N, self.N, bias=False) for i in range(K)])
        self.theta = nn.ParameterList([nn.Parameter(torch.tensor(lag / L)) for i in range(K)])

        A_t = torch.from_numpy(A).float()
        W1_ista = A_t.T / L
        W2_ista = torch.eye(self.N) - A_t.T @ A_t / L
        for k in range(K):
            self.W1[k].weight.data = W1_ista.clone()
            self.W2[k].weight.data = W2_ista.clone()

    def soft_threshold(self, z, theta):
        return torch.sign(z) * torch.clamp(torch.abs(z) - theta, 0.0)

    def forward(self, b, return_all: bool = False):
        x = torch.zeros(b.shape[0], self.N)
        x_history = []

        for k in range(self.K):
            x = self.soft_threshold(self.W1[k](b) + self.W2[k](x), self.theta[k])
            x_history.append(x)

        if return_all:
            return x_history
        return x

    def compute_nmse_db(self, X: torch.Tensor, x_true: torch.Tensor) -> float:
        mse = torch.mean(torch.norm(X - x_true, dim=1)**2)
        norm_true = torch.mean(torch.norm(x_true, dim=1)**2)
        return 10.0 * torch.log10(mse / norm_true).item()
