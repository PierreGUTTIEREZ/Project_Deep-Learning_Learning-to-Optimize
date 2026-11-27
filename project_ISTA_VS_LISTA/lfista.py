import numpy as np
import torch
import torch.nn as nn

class LFISTA(nn.Module):

    def __init__(self, A: np.ndarray, L: float, lag: float, K: int):
        super(LFISTA, self).__init__()
        self.L = L
        self.lag = lag
        self.K = K
        M, N = A.shape
        self.M = M
        self.N = N

        self.W1 = nn.ModuleList([nn.Linear(self.M, self.N, bias=False) for i in range(K)])
        self.W2 = nn.ModuleList([nn.Linear(self.N, self.N, bias=False) for i in range(K)])
        self.W3 = nn.ModuleList([nn.Linear(self.N, self.N, bias=False) for i in range(K)])
        self.theta = nn.ParameterList([nn.Parameter(torch.tensor(lag / L)) for i in range(K)])

        t = [1.0]
        for k in range(K):
            t.append(self.t_k(t[-1]))
        self.beta = [0.0] + [self.Beta_k(t[k - 1], t[k]) for k in range(1, K)]  

        A_t = torch.from_numpy(A).float()
        W1_fista = A_t.T / L
        W2_fista = torch.eye(self.N) - A_t.T @ A_t / L
        for k in range(K):
            self.W1[k].weight.data = W1_fista.clone()
            self.W2[k].weight.data = (1 + self.beta[k]) * W2_fista.clone()
            self.W3[k].weight.data = -self.beta[k] * W2_fista.clone()

    def soft_threshold(self, z, theta):
        return torch.sign(z) * torch.clamp(torch.abs(z) - theta, 0.0)

    def t_k(self, t_prev):
        return float((1 + np.sqrt(1 + 4 * t_prev**2)) / 2)

    def Beta_k(self, tk_prev, tk):
        return (tk_prev - 1) / tk

    def forward(self, b, return_all: bool = False):
        x = torch.zeros(b.shape[0], self.N)
        x_prev = torch.zeros(b.shape[0], self.N)
        x_history = []

        for k in range(self.K):
            x_new = self.soft_threshold(self.W1[k](b) + self.W2[k](x) + self.W3[k](x_prev), self.theta[k])
            x_prev, x = x, x_new
            x_history.append(x)

        if return_all:
            return x_history
        return x

    def compute_nmse_db(self, X: torch.Tensor, x_true: torch.Tensor) -> float:
        mse = torch.mean(torch.norm(X - x_true, dim=1)**2)
        norm_true = torch.mean(torch.norm(x_true, dim=1)**2)
        return 10.0 * torch.log10(mse / norm_true).item()
