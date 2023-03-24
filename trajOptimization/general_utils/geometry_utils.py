import numpy as np

def point2line_dist(lp1, lp2, p):
    vec1 = lp1-p
    vec2 = lp2-p
    vec3 = lp2-lp1
    return np.linalg.norm(np.cross(vec1,vec2)) / np.linalg.norm(vec3)

def point2line_project(lp1, lp2, p):
    vec1 = p-lp1
    vec2 = lp2-lp1
    t = np.sum(vec1*vec2) / (np.linalg.norm(vec2)**2)
    return lp1 + t*vec2

def point2segment_dist(sp1, sp2, p):
    vec1 = p-sp1
    vec2 = sp2-sp1
    vec3 = p-sp2
    seg_length = np.linalg.norm(vec2)
    r = np.sum(vec1*vec2) / (seg_length*seg_length)
    if r <= 0:
        return np.linalg.norm(vec1)
    elif r >= 1:
        return np.linalg.norm(vec3)
    else:
        return np.linalg.norm(np.cross(vec1,vec3)) / seg_length


def polyder(t, k: int=0, order: int=5):
    """
    t (float): time
    k (int): order of derivative
    order (int): order of the polynomial
    The input describes a (d/dt^k)(1+t+t^2+...+t^(order)) = (d/dt^k)[1 1 ... 1] = [c_0 c_1 c_2 .. c_(order-k)]
    Returns [0 0 .. c_0 c_1*t c_2*t^2 .. c_(order-k)*t^(order-k)], size=(order+1,)

    Notice [matrix product]: if p(t) = [1 t t^2 .. t^n]*p, 
        then v(t) = polyder(t, 1, order)*p,
        a(t) = polyder(t, 2, order)*p
        jerk(t) = polyder(t, 3, order)*p
        snap(t) = polyder(t, 4, order)*p
    """
    terms = np.zeros(order + 1)
    coeffs = np.polyder([1] * (order+1), k)
    coeffs = coeffs[::-1] # inverse coeffs
    pows = t**np.arange(0, order + 1 - k, 1)
    terms[k:] = coeffs*pows
    return terms

def full_polyder(t, max_k: int=4, order: int=5):
    """
    Generate a matrix of:
        [polyder(t, 1, order),
        polyder(t, 2, order),
        ...
        polyder(t, max_k, order)]
    """
    terms = np.array([polyder(t, k, order) for k in range(max_k + 1)])
    return terms

def Hessian(T, order: int=5, opt: int=4):
    """
    T (array): len(T) = segment_num, include time allocation of every trajectory segments
    order (int): order of the polynomial
    opt (int): formulation of the cost function, minimum jerk means opt = 4

    Returns Q(m(n+1)*m(n+1))
    Notice if the polynomials are represented as p_i(t) = [1 t t^2 .. t^n]*p_i, p=[p_1 p_2 .. p_m](m(n+1)*1) (1-D polynomials)
    Then the cost function is represented as J = p^T*Q*p
    """
    n = len(T)
    Q = np.zeros(((order+1) * n, (order+1) * n))
    for k in range(n):
        m = np.arange(0, opt, 1)
        for i in range(order + 1):
            for j in range(order + 1):
                if i >= opt and j >= opt:
                    pow = i + j - 2*opt + 1
                    Q[(order+1) * k + i, (order+1) * k + j] = 2 * np.prod((i-m) * (j-m)) * T[k]**pow / pow
    return Q

def Circle_waypoints(n,Tmax = 2*np.pi):
    t = np.linspace(0,Tmax, n)
    x = 1+0.5*np.cos(t)
    y = 1+0.5*np.sin(t)
    z = 1+0*t
    return np.stack((x, y, z), axis=-1)

def Helix_waypoints(n,Tmax = 2*np.pi):

    t = np.linspace(0, Tmax, n)
    x = 1+0.5*np.cos(t)
    print(x)
    y = 1+0.5*np.sin(t)
    z = t/Tmax*2
    print(np.stack((x, y, z), axis=-1))

    return np.stack((x, y, z), axis=-1)

if __name__ == '__main__':
    # test function usage
    T = [2, 5]
    print(Hessian(T, order=5, opt=4))