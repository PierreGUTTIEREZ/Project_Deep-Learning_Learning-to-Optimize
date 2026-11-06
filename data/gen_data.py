# Importations
import numpy as np

def gen_A(m : int, n : int):
    A = np.random.randn(m, n) / np.sqrt(m)
    for i in range (len(A)):
        c = A[:, i]
        c /= np.linalg.norm(c)
        A[:, i] = c
    return A

def gen_x_star(m : int):
    x = np.zeros(m)
    for v in x:
        x = np.random.binomial(1, 0.1) * np.random.randn(1)
    return x

def gen_B(A : np.array, x_star : np.array, e : np.array = None):
    return A @ x_star + e

