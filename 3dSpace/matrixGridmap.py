#!~/anaconda3/envs/teb/bin/python
import cupy as cp
from cupyx.scipy.signal import convolve
from utils.math_utils import *

# import matplotlib.pyplot as plt
# import matplotlib.colors as mpc
# import matplotlib.image as mpimg

class OccupancyGridMap3D:

    def __init__(self, center_crd=[0,0,0], half_extend=[10.,10.,10.], cell_size=.03, occupancy_threshold=.5):
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
        self.half_dim = cp.floor_divide(cp.array(half_extend), self.cell_size)
        self.half_size = self.cell_size * self.half_dim
        self.size = 2 * self.half_size
        self.dim = 2 * self.half_dim
        self._precheck()
        self.data = cp.zeros((self.dim[0], self.dim[1], self.dim[2]), dtype=bool)
        self.start_crd = self.center - self.half_size
        self.obj_info = {}
    
    def _precheck(self):
        for i in range(3):
            if self.size[i] <= 0 or self.dim[i] <= 0:
                print("Invalid initialization parameters!")
                exit(-1)
    
    def get_center(self):
        return cp.asnumpy(self.center)
    
    def get_size(self):
        return cp.asnumpy(self.size)
    
    def get_dim(self):
        return cp.asnumpy(self.dim)

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
            if lcrd[i] < 0 or lcrd[i] > self.size[i]:
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
            {type: "height_field", start_pos_bias(1x3 array), height_field(object)}
        """
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
                start_pos_bias = cp.array(param['start_pos_bias'])
                height_field = param['height_field']
                start_pos = cp.array(height_field.get_start_pos()) + start_pos_bias
                data = np.array(param['height_data'])
                if only_height_field is None:
                    terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                        meshScale=param['scale'],
                                                        heightfieldTextureScaling=(data.shape[0]-1)/2,
                                                        heightfieldData=param['height_data'].flatten(order='F'),
                                                        numHeightfieldRows=data.shape[0],
                                                        numHeightfieldColumns=data.shape[1],
                                                        physicsClientId=pclient)
                    only_height_field = terrainShape
                else:
                    terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                        meshScale=param['scale'],
                                                        heightfieldTextureScaling=(data.shape[0]-1)/2,
                                                        heightfieldData=param['height_data'],
                                                        numHeightfieldRows=data.shape[0],
                                                        numHeightfieldColumns=data.shape[1],
                                                        replaceHeightfieldIndex=only_height_field,
                                                        physicsClientId=pclient)
                    continue
                uid = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=terrainShape,
                                    physicsClientId=pclient)
                pos_bias = [data.shape[0]*param['scale'][0] / 2,
                            data.shape[1]*param['scale'][1] / 2,
                            0]
                p.resetBasePositionAndOrientation(uid, pos_bias, [0,0,0,1], physicsClientId=pclient)
                p.changeVisualShape(uid, -1, rgbaColor=[1,1,1,1], physicsClientId=pclient)
            else:
                print("Warning: undefined obstacle type! Add nothing!")

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
        half_len = max(1, int(round(dist/self.cell_size)))

        conv_core = np.ones((2*half_len+1, 2*half_len+1, 2*half_len+1))
        new_data = signal.convolve(self.data, conv_core, mode='same')
        self.data = new_data
        self.map_normalize()
        
    def plot(self, axis=None, grid=False):
        """
        plot the grid map
        """
        if axis == None:
            figure = plt.figure('grid_map')
            axis = figure.add_axes([0.05,0.05,0.95,0.95], projection='3d')
        ax = axis
        if grid:
            color = mpc.to_rgba('lightseagreen', alpha=0.4)
            vs = self.data
            x,y,z = np.indices(np.array(vs.shape)+1)
            ax.voxels(x*self.cell_size, y*self.cell_size, z*self.cell_size, vs, facecolors=color)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('3D Voxel Map')
        else:
            print("Warning: This function is only suitable for drawing the obstacles that are not hollowed out!")
            # true_data = np.clip(np.around(self.data), 0., 1.)
            Z = self.data.sum(axis=2) * self.cell_size
            X, Y = np.meshgrid(np.linspace(0,self.dim_meters[0], num=self.dim_cells[0]),
                                np.linspace(0,self.dim_meters[1], num=self.dim_cells[1]))
            ax.contour(X, Y, Z.T, 3, extend3d=True, colors='darkgrey', alpha=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('3D Contour')

        plt.show(block=False)
        plt.pause(5)
        plt.close()

class GridMapPath(OccupancyGridMap3D):

    def __init__(self, array3D, crd_bias=[0,0,0], cell_size=.1, occupancy_threshold=.5):
        super().__init__(array3D, crd_bias, cell_size, occupancy_threshold)
        self.visited = np.zeros(self.dim_cells, dtype=np.float32)
    
    def get_data_idx(self, point_idx):
        x_index, y_index, z_index = point_idx
        if x_index < 0 or y_index < 0 or z_index < 0 \
            or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1] or z_index >= self.dim_cells[2]:
            raise Exception('Point is outside map boundary')
        return self.data[x_index][y_index][z_index]

    def mark_visited_idx(self, point_idx):
        x_index, y_index, z_index = point_idx
        if x_index < 0 or y_index < 0 or z_index < 0 \
            or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1] or z_index >= self.dim_cells[2]:
            raise Exception('Point is outside map boundary')
        self.visited[x_index][y_index][z_index] = 1.0

    def is_visited_idx(self, point_idx):
        x_index, y_index, z_index = point_idx
        if x_index < 0 or y_index < 0 or z_index < 0 \
            or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1] or z_index >= self.dim_cells[2]:
            raise Exception('Point is outside map boundary')
        if self.visited[x_index][y_index][z_index] == 1.0:
            return True
        else:
            return False

def GridMapFromImage(imgpath: str, height: float, cell_size: float=0.1, with_path: bool=True):
    img_array = mpimg.imread(imgpath)
    gray_img = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            if gray_img[i][j] >= 0.5:
                gray_img[i][j] = 1
            else:
                gray_img[i][j] = 0
    height_num = int(round(height / cell_size))
    _data = np.broadcast_to(gray_img.reshape((gray_img.shape[0], gray_img.shape[1], 1)),
                                (gray_img.shape[0], gray_img.shape[1], height_num))
    if not with_path:
        return OccupancyGridMap3D(_data.copy(), cell_size=cell_size)
    else:
        return GridMapPath(_data.copy(), cell_size=cell_size)

def BlankGridMap(length: float, width: float, height: float, cell_size: float=0.05, with_path: bool=True):
    len_dim = int(round(length / cell_size))
    wid_dim = int(round(width / cell_size))
    hgt_dim = int(round(height / cell_size))
    _data = np.zeros((len_dim,wid_dim,hgt_dim), dtype=int)
    if not with_path:
        return OccupancyGridMap3D(_data.copy(), cell_size=cell_size)
    else:
        return GridMapPath(_data.copy(), cell_size=cell_size)

if __name__ == '__main__':
    import os
    # import numpy as np

    # path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/pictures/test.png'
    # grid = GridMapFromImage(path, 3.)
    # grid.obstacle_dilation(0.2)
    # grid.plot(grid=False)

    a = cp.array([[0,1,1,0], [0,0,0,1]], dtype=bool)
    b = (0,1)
    # print(convolve(a, b, mode='same'))
    print(a[b])