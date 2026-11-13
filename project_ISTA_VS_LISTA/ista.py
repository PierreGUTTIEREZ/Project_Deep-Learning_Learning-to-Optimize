# Classe ISTA fait à la main

import numpy as np
from math import max

class ISTA():
    def __init__(self, A : np.array, b : np.array , L : float , lag : float):
        self.A=A
        self.b=b
        self.L= L
        self.lag=lag


    def step_ista(self, xk : np.array):
        z= xk - (1/self.L) * np.linalg.matrix_transpose(self.A) @ (self.A@xk -self.b)
        xk= np.sign(z)* max(np.linalg.norm(z)-self.lag/self.L,0)
        return xk

    def forward(self,x):
        while np.linalg.norm(self.A ):
            x=self.step_ista(x)


