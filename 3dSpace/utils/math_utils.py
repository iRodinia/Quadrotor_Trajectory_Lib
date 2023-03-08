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

def eulerAngles2rotMatrix(angles):
    """Calculates Rotation Matrix from given euler angles.
    
    Z-Y-X defined RPY Euler angles [rx, ry, rz]
    Rotate rx along X, ry along Y, rz along Z sequentially
    
    :param angles: 1-by-3 list [rx, ry, rz] angle in radius
    :return:
    rotMat
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angles[0]), -np.sin(angles[0])],
                    [0, np.sin(angles[0]), np.cos(angles[0])]
                    ])
 
    R_y = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                    [0, 1, 0],
                    [-np.sin(angles[1]), 0, np.cos(angles[1])]
                    ])
 
    R_z = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                    [np.sin(angles[2]), np.cos(angles[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
