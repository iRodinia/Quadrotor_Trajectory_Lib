#!~/anaconda3/envs/teb/bin/python
import cupy as cp
from cupyx.scipy import ndimage
from utils.math_utils import *

class matrixGridMap3D:

    def __init__(self, center_crd=[0,0,0], half_extend=[5.,5.,3.], cell_size=.03, occupancy_threshold=.5):
        """ Creates a grid map

        :param center_crd: world coordinate of the center of the grid map
        :param half_extend: half size of the detection area. May not be same as the actual value used
        :param cell_size: discretize accuracy, in meters
        :param occupancy_threshold: A threshold to determine whether a cell is occupied or free.
            A cell is considered occupied if its value >= occupancy_threshold, free otherwise.
        """
        self.center = cp.array(center_crd)
        self.cell_size = cell_size
        self.occupancy_threshold = occupancy_threshold
        self.half_dim_shape = cp.floor_divide(cp.array(half_extend), self.cell_size)
        self.half_shape = self.cell_size * self.half_dim_shape
        self.shape = 2 * self.half_shape
        self.dim_shape = 2 * self.half_dim_shape.get().astype(int)
        self._precheck()
        self.data = cp.zeros((self.dim_shape[0], self.dim_shape[1], self.dim_shape[2]), dtype=bool)
        self.start_crd = self.center - self.half_shape
        self.obj_info = {}
        self.dilation = 0.
    
    def _precheck(self):
        for i in range(3):
            if self.shape[i] <= 0 or self.dim_shape[i] <= 0:
                print("Invalid initialization parameters!")
                exit(-1)
    
    def get_center(self):
        return cp.asnumpy(self.center)
    
    def get_size(self):
        return cp.asnumpy(self.shape)
    
    def get_dim(self):
        return cp.asnumpy(self.dim_shape)
    
    def get_data(self):
        return cp.asnumpy(self.data)

    def _wcrd_to_lcrd(self, wcrd):
        """Convert a world coordinate to local coordinate

        :param wcrd (3, array): world coordinate
        
        Return:
        local coordinate (3, array)
        """
        return cp.array(wcrd) - self.start_crd
    
    def _lcrd_to_wcrd(self, lcrd):
        """Convert a local coordinate to world coordinate

        :param lcrd (3, array): local coordinate
        
        Return:
        world coordinate (3, array)
        """
        return cp.array(lcrd) + self.start_crd
    
    def _is_inside_local(self, lcrd):
        """If a local coordinate inside the map

        :param lcrd (3, array): local coordinate
        
        Return:
        inside (bool)
        """
        for i in range(3):
            if lcrd[i] < 0 or lcrd[i] > self.shape[i]:
                return True
        return False
    
    def is_inside(self, wcrd):
        """If a world coordinate inside the map

        :param wcrd (3, array): world coordinate
        
        Return:
        inside (bool)
        """
        lcrd = self._wcrd_to_lcrd(wcrd)
        return self._is_inside_local(lcrd)
    
    def _lcrd_to_idx(self, lcrd):
        """Convert a local coordinate to local matrix index

        :param lcrd (3, array): local coordinate
        
        Return:
        index (3, array, int)
        """
        if self._is_inside_local(lcrd):
            print('Invalid lookup coordinate!')
            return None
        idx = cp.floor_divide(cp.array(lcrd), self.cell_size)
        for i in range(3):
            if idx[i] == self.dim_cells[i]:
                idx[i] -= 1
        return (idx[0], idx[1], idx[2])
    
    def _idx_to_lcrd(self, idx):
        """Convert a local matrix index to local coordinate

        :param idx (3, tuple): local matrix index
        
        Return:
        crd (3, array)
        """
        lb_corner = cp.array([idx[0], idx[1], idx[2]]) * self.cell_size
        center_bias = cp.array([.5, .5, .5]) * self.cell_size
        return lb_corner + center_bias
    
    def _wcrd_to_idx(self, wcrd):
        lcrd = self._wcrd_to_lcrd(wcrd)
        return self._lcrd_to_idx(lcrd)
    
    def _idx_to_wcrd(self, idx):
        lcrd = self._idx_to_lcrd(idx)
        return self._lcrd_to_wcrd(lcrd)

    def is_occupied(self, wcrd):
        """If a world coordinate is occupied by obstacles

        :param wcrd (3, array): world coordinate
        
        Return:
        occupied (bool)
        """
        idx = self._wcrd_to_idx(wcrd)
        if self.data[idx] >= self.occupancy_threshold:
            return True
        else:
            return False
    
    def map_normalize(self):
        self.data.astype(bool)

    def _refresh_obstacles(self):
        """Load obstacles to the map according to self.obj_dict.

            Supported items:
            {type: "box", center(1x3 array), half_extend(1x3 array), yaw_rad}
            {type: "cylinder", center(1x3 array), radius, height}
            {type: "ball", center(1x3 array), radius}
            {type: "height_field", height_field(object)}
        """
        self.data = cp.zeros((self.dim_shape[0], self.dim_shape[1], self.dim_shape[2]), dtype=bool)
        for (_, param) in self.obj_info.items():
            if param['type'] == "box":
                rotmat = cp.array(eulerAngles2rotMatrix([0., 0., param['yaw_rad']]))
                box_center = cp.array(param['center'])
                half_extend = cp.array(param['half_extend'])
                xmin = - half_extend[0]
                xmax = half_extend[0]
                ymin = - half_extend[1]
                ymax = half_extend[1]
                zmin = - half_extend[2]
                zmax = half_extend[2]
                for i in range(xmin, xmax, self.cell_size):
                    for j in range(ymin, ymax, self.cell_size):
                        for k in range(zmin, zmax, self.cell_size):
                            rot_vec = rotmat.dot(cp.array([i,j,k]).T)
                            vec = rot_vec + box_center
                            if self.is_inside(vec):
                                idx = self._wcrd_to_idx(vec)
                                self.data[idx] = 1

            elif param['type'] == "cylinder":
                cld_center = cp.array(param['center'])
                radius = param['radius']
                height = param['height']
                hmin = - height / 2
                hmax = height / 2
                for i in range(hmin, hmax, self.cell_size):
                    for j in range(0, radius, self.cell_size):
                        for k in cp.linspace(0, 2*cp.pi, 180, endpoint=False):
                            x = j * cp.cos(k)
                            y = j * cp.sin(k)
                            z = i
                            vec = cp.array([x, y, z]) + cld_center
                            if self.is_inside(vec):
                                idx = self._wcrd_to_idx(vec)
                                self.data[idx] = 1

            elif param['type'] == 'ball':
                ball_center = param['center']
                radius = param['radius']
                for i in range(0, radius, self.cell_size):
                    for j in cp.linspace(0, 2*cp.pi, 180, endpoint=False):
                        for k in cp.linspace(0, cp.pi, 91):
                            x = i * cp.sin(k) * cp.cos(j)
                            y = i * cp.sin(k) * cp.sin(j)
                            z = i * cp.cos(k)
                            vec = cp.array([x, y, z]) + ball_center
                            if self.is_inside(vec):
                                idx = self._wcrd_to_idx(vec)
                                self.data[idx] = 1

            elif param['type'] == "height_field":
                height_field = param['height_field']
                for i in range(self.dim_shape[0]):
                    for j in range(self.dim_shape[1]):
                        wcrd = self._idx_to_wcrd((i,j,0))
                        if not height_field.is_pos_out_range([wcrd[0], wcrd[1]]):
                            h = height_field.get_height([wcrd[0], wcrd[1]]).get()
                            h_dim = int(round(h/self.cell_size))
                            for k in range(min(h_dim, self.dim_shape[2])):
                                self.data[i,j,k] = 1
                
            else:
                print("Warning: undefined obstacle type! Add nothing!")
        
        self.obstacle_dilation(dist=self.dilation)

    def add_obstacles(self, obs_dict: dict, mode=0):
        """Add obstacles to the map

        :param obs_dict: self-defined obstacle dictionary. Supported items are:
            {type: "box", center(1x3 array), half_extend(1x3 array), yaw_rad, fixed(bool)}
            {type: "cylinder", center(1x3 array), radius, height, fixed(bool)}
            {type: "ball", center(1x3 array), radius, fixed(bool)}
            {type: "height_field", scale(1x3 array), grid_size(1x2 array), height_data(MxN)}(seperatly loaded)
        :param mode: mode of how to add obstacles.
            0: add to existing obstacles
            1: change the existing obstacles to obs_dict
        """
        if mode == 0:
            self.obj_info.update(obs_dict)
        elif mode == 1:
            self.obj_info = obs_dict
        else:
            print("Unsupported obstacle refresh mode!")
        self._refresh_obstacles()

    def obstacle_dilation(self, dist=0.):
        half_len = int(round(dist/self.cell_size))
        if half_len == 0:
            return
        core = ndimage.generate_binary_structure(3, 1)
        if half_len >= 2:
            core = ndimage.iterate_structure(core, half_len)
        self.data = ndimage.binary_dilation(self.data, structure=cp.array(core))
        self.dilation = dist
        

if __name__ == '__main__':
    import os
    import time
    from heightField import constructHeightFieldFromImg

    path = os.path.abspath(os.path.dirname(__file__)) + '/pictures/test.png'
    field = constructHeightFieldFromImg(path, 3.0, [0,0], cell_size=0.03, data_shape=(60,70))
    obs_dict = {'obs1': {'type': 'height_field', 'height_field':field}}

    core = ndimage.generate_binary_structure(3, 1)
    core = ndimage.iterate_structure(core, 3)



    # start_time = time.time()
    # map = matrixGridMap3D(center_crd=[0,0,0], half_extend=[3,3,2], cell_size=0.05)
    # cons_time = time.time()
    # print('construction time: ', cons_time - start_time)
    # map.add_obstacles(obs_dict=obs_dict)
    # load_time = time.time()
    # print('load time: ', load_time - cons_time)
    # map.obstacle_dilation(dist=0.5)
    # print('dialation time: ', time.time() - load_time)