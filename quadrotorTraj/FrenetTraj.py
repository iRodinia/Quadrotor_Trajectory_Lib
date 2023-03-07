import numpy as np
from scipy import interpolate
from utils.math_utils import *

class FrenetTraj:
    def __init__(self, trajectory, accuracy=0.01):
        self.traj = trajectory
        self.discrete_accuracy = accuracy
        for i in range(4):
            if not self.traj.is_smooth(i):
                print("Warning: Trajectory not continue at ", i, "th derivative! May cause large error!")
                break
        
        self.timestamps = np.arange(0, trajectory.duration, self.discrete_accuracy)
        self.duration = self.discrete_accuracy * (self.timestamps.size-1)
        self.key_points = np.array([self.traj.eval(t) for t in self.timestamps])
        pos_err = np.diff(self.key_points[:,:3], axis=0)
        dist_err = np.concatenate(([0.], [np.linalg.norm(d) for d in pos_err]))
        self.key_length = np.cumsum(dist_err)
        self.length_fnc = interpolate.interp1d(self.timestamps, self.key_length, kind='linear')

        traj_pieces = self.traj.get_pieces()
        self.derivative1 = [p.derivative() for p in traj_pieces]
        self.derivative2 = [p.derivative() for p in self.derivative1]
        self.derivative3 = [p.derivative() for p in self.derivative2]

        self.key_curv = np.array([self.eval_curvature(t) for t in self.timestamps])
        self.key_tors = np.array([self.eval_torsion(t) for t in self.timestamps])

    def get_coefficients(self):
        return self.traj.get_coef_mat()

    def eval(self, t, maximum_allowed=True):
        return self.traj.eval(t, maximum_allowed=maximum_allowed)
    
    def eval_flat_states(self, t, maximum_allowed=True):
        return self.traj.eval_flat_states(t, maximum_allowed=maximum_allowed)
    
    def eval_length(self, t, maximum_allowed=True):
        assert t >= 0
        if maximum_allowed:
            t = min(self.duration, t)
        else:
            assert t <= self.duration
        return self.length_fnc(t)
    
    def eval_curvature(self, t, maximum_allowed=True):
        assert t >= 0
        if maximum_allowed:
            t = min(self.duration, t)
        else:
            assert t <= self.duration
        pidx = self.traj.time_at_which_piece(t)
        prev_time = self.traj.total_time_before_n_piece(pidx)
        pd1 = self.derivative1[pidx].eval(t-prev_time, maximum_allowed=False)[:3]
        pd2 = self.derivative2[pidx].eval(t-prev_time, maximum_allowed=False)[:3]
        mid_vec = np.cross(pd1, pd2)
        curvature = default_divide(np.linalg.norm(mid_vec), np.linalg.norm(pd1)**3)
        return curvature
    
    def eval_torsion(self, t, maximum_allowed=True):
        assert t >= 0
        if maximum_allowed:
            t = min(self.duration, t)
        else:
            assert t <= self.duration
        pidx = self.traj.time_at_which_piece(t)
        prev_time = self.traj.total_time_before_n_piece(pidx)
        pd1 = self.derivative1[pidx].eval(t-prev_time, maximum_allowed=False)[:3]
        pd2 = self.derivative2[pidx].eval(t-prev_time, maximum_allowed=False)[:3]
        pd3 = self.derivative3[pidx].eval(t-prev_time, maximum_allowed=False)[:3]
        mid_vec = np.cross(pd1, pd2)
        torsion = default_divide(np.dot(mid_vec, pd3), np.linalg.norm(mid_vec)**2)
        return torsion
    
    def eval_frame_axis(self, t, maximum_allowed=True):
        """compute the Frenet frame vector at a given t
        Args:
            t: The evaluate parameter
        Return:
            T: tangent vector
            N: normal vector
            B: subnormal vector
        """
        assert t >= 0
        if maximum_allowed:
            t = min(self.duration, t)
        else:
            assert t <= self.duration
        pidx = self.traj.time_at_which_piece(t)
        prev_time = self.traj.total_time_before_n_piece(pidx)
        pd1 = self.derivative1[pidx].eval(t-prev_time, maximum_allowed=False)[:3]
        pd2 = self.derivative2[pidx].eval(t-prev_time, maximum_allowed=False)[:3]
        T = pd1 / np.linalg.norm(pd1)
        B = np.cross(pd1, pd2) / np.linalg.norm(np.cross(pd1, pd2))
        N = np.cross(B, T)
        return T, N, B

    def closest_key_pos_idx(self, crd, ref_pos=None):
        """compute the closest key position index from a cartesian coordinate [x,y,z]
        Args:
            crd (1*3 array): The cartesian coordinate
            ref_pos (float > 0): The current reference position for more accurate evaluation
        Return:
            index
        """
        min_dist = np.inf
        min_index = -1
        for i in range(self.key_points.shape[0]):
            dist = np.linalg.norm(crd - self.key_points[i,:3])
            if dist <= min_dist:
                min_index = i 
                min_dist = dist
        if min_index != 0 and min_dist != i:
            if min_dist*self.key_curv[min_index] > 1:
                print("Bias too large! Frenet coordinates not applicable")
        return min_index

    def closest_point_from_cartesian(self, crd, ref_pos=None):
        """compute the closest position on the curve from a cartesian coordinate [x,y,z]

        The head and tail of the curve are extended by tangent lines

        Args:
            crd (1*3 array): The cartesian coordinate
            ref_time (float > 0): The current time for more accurate evaluation
        Return:
            t: the closest point variable t
            pos (1*3 array): The closest position coordinate
        """
        idx = self.closest_key_pos_idx(crd, ref_pos=ref_pos)
        if idx == 0:
            v = self.key_points[idx+1,:3] - self.key_points[idx,:3]
            vp = crd - self.key_points[idx,:3]
            vec_proj = vec_project(vp, v)
            pos = self.key_points[idx,:3] + vec_proj*v/np.linalg.norm(v)
            t = 0
        elif idx == self.key_points.shape[0]-1:
            v = self.key_points[idx-1,:3] - self.key_points[idx,:3]
            vp = crd - self.key_points[idx,:3]
            vec_proj = vec_project(vp, v)
            pos = self.key_points[idx,:3] + vec_proj*v/np.linalg.norm(v)
            t = self.duration
        else:
            v1 = self.key_points[idx-1,:3] - self.key_points[idx,:3]
            v2 = self.key_points[idx+1,:3] - self.key_points[idx,:3]
            vp = crd - self.key_points[idx,:3]
            vec_proj1 = vec_project(vp, v1)
            vec_proj2 = vec_project(vp, v2)
            if vec_proj1 >= 0:
                pos = self.key_points[idx,:3] + vec_proj1*v1/np.linalg.norm(v1)
                t = self.timestamps[idx] - vec_proj1/np.linalg.norm(vp) * self.discrete_accuracy
            else:
                pos = self.key_points[idx,:3] + vec_proj2*v2/np.linalg.norm(v2)
                t = self.timestamps[idx] + vec_proj2/np.linalg.norm(vp) * self.discrete_accuracy
        return t, pos
    
    def cartesian_to_frenet(self, crd):
        """compute the frenet coordinate from a cartesian coordinate [x,y,z]

        The head and tail of the curve are extended by tangent lines

        Args:
            crd (1*3 array): The cartesian coordinate
        Return:
            s: arc length
            l: N coordinate
            d: B coordinate
        """
        t, pos = self.closest_point_from_cartesian(crd)
        _, N, B = self.eval_frame_axis(t)
        vp = crd - pos
        l = np.dot(N, vp)
        d = np.dot(B, vp)
        if t == 0:
            s = -np.linalg.norm(pos-self.key_points[0,:3])
        elif t == self.duration:
            s = self.eval_length(self.duration) + np.linalg.norm(pos-self.key_points[-1,:3])
        else:
            s = self.eval_length(t)
        return s, l, d
