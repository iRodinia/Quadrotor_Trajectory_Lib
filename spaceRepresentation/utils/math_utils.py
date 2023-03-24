"""An analytical geometry module utilized by zkz's quadrotor trajectory library.

All vectors and matrixes used are assumed to be Numpy.array.

"""

import numpy as np
from scipy.interpolate import interp1d

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

def resize_vector(vec, new_len: int):
    assert new_len > 0
    length = len(vec)
    output = np.zeros(new_len)
    f = interp1d(range(length), vec, kind='linear')
    for i in range(new_len):
        x = i * (length-1) / (new_len-1)
        output[i] = f(x)
    return output

def resize_map(map, new_size: tuple):
    assert len(new_size) == 2
    assert new_size[0] > 0 and new_size[1] > 0
    map = np.array(map)
    size = map.shape
    temp = np.zeros((size[0], new_size[1]))
    for i in range(size[0]):
        temp[i,:] = resize_vector(map[i,:], new_size[1])
    output = np.zeros((new_size[0], new_size[1]))
    for j in range(new_size[0]):
        output[:,j] = resize_vector(temp[:,j], new_size[0])
    return output