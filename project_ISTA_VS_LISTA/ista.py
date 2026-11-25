import numpy as np

class ISTA:
    def __init__(self, A: np.ndarray, b: np.ndarray, L: float, lag: float, x_true: np.ndarray = None):
        """
        A      : Matrice de mesure (M, N)
        b      : Observations (B, M)
        L      : Constante de Lipschitz = ||A||_2^2
        lag    : Paramètre de régularisation lambda
        x_true : Vérité terrain (B, N) pour le suivi de la NMSE
        """
        self.A = A
        self.b = b
        self.L = L
        self.lag = lag
        self.theta = lag / L
        self.x_true = x_true

    def step_ista(self, X: np.ndarray) -> np.ndarray:
        residual = X @ self.A.T - self.b
        Z = X - (1.0 / self.L) * (residual @ self.A)
        X_next = np.sign(Z) * np.maximum(np.abs(Z) - self.theta, 0.0)
        return X_next

    def compute_nmse_db(self, X: np.ndarray) -> float:
        if self.x_true is None:
            return 10.0 * np.log10(np.mean(np.linalg.norm(self.b - X @ self.A.T, axis=1)**2))
        
        # NMSE standard en dB
        mse = np.mean(np.linalg.norm(X - self.x_true, axis=1)**2)
        norm_true = np.mean(np.linalg.norm(self.x_true, axis=1)**2)
        return 10.0 * np.log10(mse / norm_true)

    def forward(self, X_init: np.ndarray, epochs: int):
        X = X_init.copy()
        nmse_history = [self.compute_nmse_db(X)]

        for _ in range(epochs):
            X = self.step_ista(X)
            nmse_history.append(self.compute_nmse_db(X))

        return X, nmse_history