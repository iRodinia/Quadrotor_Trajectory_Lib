import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

import numpy as np
from scipy import optimize
from quadrotorTraj.basicPolyTraj import PolyTrajectory
from .general_utils.geometry_utils import *

class MinimumSnapOptimizer:

    def __init__(self, pathpoints: np.ndarray, avg_vel: float=2., gamma: float=1., poly_order: int=5,
                 continuity_order: int=2, start_derivatives: np.ndarray=None, end_derivatives: np.ndarray=None):
        # pathpoints is in format: [p1 p2 p3 ...]
        # gamma is only used in the minimum snap with time allocation
        assert pathpoints.ndim == 2 and avg_vel > 0
        assert poly_order >= 5, "At least 5-th order polynomial is required for continuity up to acceleration!"
        
        self.order = int(poly_order)
        self.continuity_order = int(min(continuity_order, (poly_order-1)/2))
        self.points_len, self.dim = pathpoints.shape
        if poly_order % 2 != 0:
            self.start_constraint_num = int((poly_order+1)/2)
            self.end_constraint_num = int((poly_order+1)/2)
        else:
            self.start_constraint_num = int(poly_order/2+1)
            self.end_constraint_num = int(poly_order/2)

        self.start_derivatives = np.zeros((self.start_constraint_num-1, self.dim))
        if start_derivatives != None:
            for i in range(min(start_derivatives.shape[0], self.start_constraint_num-1)):
                self.start_derivatives[i,:] = start_derivatives[i,:]
        self.end_derivatives = np.zeros((self.end_constraint_num-1, self.dim))
        if end_derivatives != None:
            for i in range(min(end_derivatives.shape[0], self.end_constraint_num-1)):
                self.end_derivatives[i,:] = end_derivatives[i,:]

        self.pathpoints = pathpoints
        diff = self.pathpoints[0:-1] - self.pathpoints[1:]
        self.init_T = np.linalg.norm(diff, axis=-1) / avg_vel
        self.gamma = gamma


    def _get_constraints(self, T):
        # T is the time allocation array
        # T = [t1 t2 ... tm], ti is the running time of the i-th segment
        # continuity_order is the order of continuity at every midpoints
        n_seg = self.points_len - 1
        n_coef = self.order + 1
        known_constraints_num = self.start_constraint_num + self.end_constraint_num + (self.continuity_order+2)*(n_seg-1)
        free_derivatives_num = n_seg*n_coef - known_constraints_num

        A_cons = np.zeros((known_constraints_num, n_seg*n_coef))
        B_cons = np.zeros((known_constraints_num, self.dim))
        A_free = np.zeros((free_derivatives_num, n_seg*n_coef))

        # start and end constrains
        A_cons[:self.start_constraint_num, :n_coef] = full_polyder(0., max_k=self.start_constraint_num-1, order=self.order)
        B_cons[0] = self.pathpoints[0]
        B_cons[1:self.start_constraint_num, :] = self.start_derivatives
        A_cons[self.start_constraint_num:self.start_constraint_num+self.end_constraint_num, -n_coef:] = \
            full_polyder(T[-1], max_k=self.end_constraint_num-1, order=self.order)
        B_cons[self.start_constraint_num] = self.pathpoints[-1]
        B_cons[self.start_constraint_num+1:self.start_constraint_num+self.end_constraint_num, :] = self.end_derivatives
        
        # midpoint constraints
        idx1 = self.start_constraint_num + self.end_constraint_num
        for i in range(n_seg-1):
            A_cons[idx1 + i, n_coef*i:n_coef*(i+1)] = polyder(T[i], k=0, order=self.order) # set as t=T constraints
            B_cons[idx1 + i, :] = self.pathpoints[i+1, :]

        # continuity constraints
        idx2 = idx1 + n_seg - 1
        for i in range(n_seg-1):
            A_cons[idx2+i*(self.continuity_order+1):idx2+(i+1)*(self.continuity_order+1), n_coef*i:n_coef*(i+1)] =\
                full_polyder(T[i], max_k=self.continuity_order, order=self.order)
            A_cons[idx2+i*(self.continuity_order+1):idx2+(i+1)*(self.continuity_order+1), n_coef*(i+1):n_coef*(i+2)] =\
                - full_polyder(0., max_k=self.continuity_order, order=self.order)
        
        # t=T free variables
        for i in range(n_seg-1):
            A_free[i*(self.end_constraint_num-1):(i+1)*(self.end_constraint_num-1), n_coef*i:n_coef*(i+1)] =\
                full_polyder(T[i], max_k=self.end_constraint_num-1, order=self.order)[1:]
        
        # t=0 free variables
        idx3 = (n_seg-1) * (self.end_constraint_num-1)
        if self.continuity_order != self.start_constraint_num-1:
            drow = self.start_constraint_num-self.continuity_order-1
            for i in range(n_seg-1):
                A_free[idx3+i*drow:idx3+(i+1)*drow, n_coef*(i+1):n_coef*(i+2)] =\
                    full_polyder(0., max_k=self.start_constraint_num-1, order=self.order)[self.continuity_order+1:]
        
        return A_cons, B_cons, A_free
    

    def minimum_snap(self, T, weight=None):
        # weight determines the cost weight for different axis
        if weight == None:
            weight = np.ones(self.dim)
        assert weight.shape[0] == self.dim
        weight_mat = np.diag(weight) # can be modified to specify different weight for different axis
        Q = Hessian(T, order=self.order)

        A_cons, B_cons, A_free = self._get_constraints(T)
        free_var_num = A_free.shape[0]
        if free_var_num != 0:
            A = np.vstack((A_cons, A_free))
            invA = np.linalg.inv(A)
            R = invA.T @ Q @ invA
            Rfp = R[:-free_var_num, -free_var_num:]
            Rpp = R[-free_var_num:, -free_var_num:]
            dp = - np.linalg.inv(Rpp) @ Rfp.T @ B_cons
            B = np.vstack((B_cons, dp))
            P = invA @ B
        else:
            invA = np.linalg.inv(A_cons)
            P = invA @ B_cons

        cost = np.trace(P.T @ Q @ P @ weight_mat)
        return P, cost

    def _get_cost_with_time(self, T):
        _, cost = self.minimum_snap(T)
        cost += self.gamma * np.sum(T)
        return cost

    def minimum_snap_with_time_allocation(self):
        T = optimize.minimize(self._get_cost_with_time, self.init_T, method="COBYLA", 
                              constraints= ({'type': 'ineq', 'fun': lambda T: T-self.init_T}))['x']
        P, cost = self.minimum_snap(T)
        return T, P, cost
    
    def convert_to_polynomial(self, T, P: np.ndarray):
        # P should be generated from self.minimum_snap() or self.minimum_snap_with_time_allocation()
        # P is the coefficient of the polynomials
        # P = [p_1 p_2 ... p_m]^T, p_i = [p_i_0 p_i_1 ... p_i_n]^T, p_i_j = [p_i_j_x p_i_j_y p_i_j_z ...] (p_i_j is a horizontal vector)
        # Thus P is of size m(n+1) * self.dim
        seg_num = int(self.points_len - 1)
        coef_mat = P.T
        mat = np.zeros((seg_num, 3*self.order+4))
        mat[:,0] = T
        for seg in range(seg_num):
            mat[seg,1:self.order+2] = coef_mat[0,seg*(self.order+1):(seg+1)*(self.order+1)]
            mat[seg,self.order+2:2*self.order+3] = coef_mat[1,seg*(self.order+1):(seg+1)*(self.order+1)]
            mat[seg,2*self.order+3:3*self.order+4] = coef_mat[2,seg*(self.order+1):(seg+1)*(self.order+1)]
        return PolyTrajectory(mat, self.order+1)


if __name__ == '__main__':
    test_points = [[1,3,5],
                   [4,2,4],
                   [5,6,7],
                   [3,4,9],
                   [9,9,9]]
    opt = MinimumSnapOptimizer(np.array(test_points), avg_vel=2., poly_order=7, continuity_order=4)
    T, P, cost = opt.minimum_snap_with_time_allocation()
    poly = opt.convert_to_polynomial(T, P)
    print(poly.eval(5.))