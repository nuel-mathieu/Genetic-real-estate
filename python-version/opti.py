import numpy as np
import random


def gradient(f, x, eps=[1e-3, 1, 1e-3, 1]):
    """Approximation numérique du gradient ∇f(x)"""
    grad = np.zeros_like(x)
    fx = f(*x)
    alpha = 1
    while np.linalg.norm(grad) < 1e-10:
        for k in range(len(x)):
            x_eps = x.copy()
            x_eps[k] += eps[k]
            grad[k] = (f(*x_eps) - fx) / eps[k]
        alpha += 1
    if alpha > 2:
        print(f"Gradient computed with alpha={alpha - 1}")
    return grad

def random_individual(bounds):
    return np.array([
        np.random.uniform(*bounds["M"]),
        np.random.uniform(*bounds["t_invest"]),
        np.random.uniform(*bounds["A"]),
        np.random.uniform(*bounds["I"]),
    ])

def mutate(x, bounds, MUT_RATE=0.1):
    x_new = x.copy()
    for i, (key, (low, high)) in enumerate(bounds.items()):
        if random.random() < MUT_RATE:
            scale = 0.1 * (high - low)
            x_new[i] += np.random.randn() * scale
            x_new[i] = np.clip(x_new[i], low, high)
    return x_new

def crossover(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2

def fitness(ind, net_worth):
    try:
        return net_worth(*ind)
    except Exception:
        return -1e9  # pénaliser les erreurs

