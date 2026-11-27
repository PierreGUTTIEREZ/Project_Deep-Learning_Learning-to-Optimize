import numpy as np

class FISTA :
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

    def S_teta(self, z: np.array) : 
        return np.sign(z) * np.maximum(np.abs(z) - self.theta, 0.0)

    def step_fista(self, X: np.ndarray, v: np.ndarray, t: float) :
        X_new = self.S_teta( v -1/self.L * ((v@self.A.T  - self.b)  @ self.A))
        t_new=(1+np.sqrt(1+4*t**2))/2
        v_new=X_new + (t-1)/t_new * (X_new - X)
        return X_new, v_new, t_new

    def compute_nmse_db(self, X: np.ndarray) -> float:
            if self.x_true is None:
                return 10.0 * np.log10(np.mean(np.linalg.norm(self.b - X @ self.A.T, axis=1)**2))
            
            # NMSE standard en dB
            mse = np.mean(np.linalg.norm(X - self.x_true, axis=1)**2)
            norm_true = np.mean(np.linalg.norm(self.x_true, axis=1)**2)
            return 10.0 * np.log10(mse / norm_true)

    def forward(self, X_init: np.ndarray, epochs: int):
        X = X_init.copy()
        v = X_init.copy()
        t = 1.0
        nmse_history = [self.compute_nmse_db(X)]

        for _ in range(epochs):
            X, v, t = self.step_fista(X[-1], v, t)
            nmse_history.append(self.compute_nmse_db(X))

        return X, nmse_history