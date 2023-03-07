"""An analytical geometry module utilized by zkz's quadrotor trajectory library.

All vectors and matrixes used are assumed to be Numpy.array.

"""

import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm

def default_divide(a: float, b: float):
    if b == 0:
        if a > 0: return np.Inf
        elif a == 0: return 0
        else: return -np.Inf
    else:
        return a / b

def vec_project(a, b):
    """Project 3-D vector a to 3-D vector b
    
    using a·b/|b|
    """
    return np.dot(a, b) / np.linalg.norm(b)