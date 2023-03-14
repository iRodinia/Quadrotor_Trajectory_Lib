import cupy as cp
import matplotlib.image as mpimg
from utils.math_utils import *

class heightField:
    def __init__(self, height_data, start_pos, cell_size=0.01, dynamic_scaling=True, data_shape: tuple=None):
        if data_shape == None:
            self.dim_shape = height_data.shape
        else:
            self.dim_shape = data_shape
        self.meter_shape = (self.dim_shape[0]*cell_size, self.dim_shape[1]*cell_size)
        self.start_pos = cp.array(start_pos)
        self.cell_size = cell_size
        if dynamic_scaling:
            self.height_data = cp.array(resize_map(height_data, self.dim_shape))
        else:
            self.height_data = cp.zeros(self.dim_shape)
            for i in range(min(self.dim_shape[0], height_data.shape[0])):
                for j in range(min(self.dim_shape[1], height_data.shape[1])):
                    self.height_data[i,j] = height_data[i,j]

    def get_height_data(self):
        return cp.asnumpy(self.height_data)
    
    def get_start_pos(self):
        return cp.asnumpy(self.start_pos)
    
    def get_aera_shape(self):
        return self.meter_shape
    
    def _is_lcrd_out_range(self, lcrd):
        if lcrd[0] < 0 or lcrd[0] > self.meter_shape[0] or lcrd[1] < 0 or lcrd[1] > self.meter_shape[1]:
            return True
        return False

    def _lcrd_to_idx(self, lcrd):
        if not self._is_lcrd_out_range(lcrd):
            x_idx = int(lcrd[0] / self.cell_size) if lcrd[0] < self.meter_shape[0] else self.cell_size[0]-1
            y_idx = int(lcrd[1] / self.cell_size) if lcrd[1] < self.meter_shape[1] else self.cell_size[1]-1
            return (x_idx, y_idx)
        else:
            print("Index out of range!")
            exit(-1)
    
    def _wcrd_to_lcrd(self, wcrd):
        w = cp.array(wcrd)
        return w - self.start_pos

    def is_pos_out_range(self, wcrd):
        lcrd = self._wcrd_to_lcrd(wcrd)
        return self._is_lcrd_out_range(lcrd)
    
    def get_height(self, wcrd):
        lcrd = self._wcrd_to_lcrd(wcrd)
        if not self._is_lcrd_out_range(lcrd):
            idx = self._lcrd_to_idx(lcrd)
            return self.height_data[idx]
        else:
            print("Warning: query position out of range!")
            return 0.
    
    def generate_height_matrix(self, from_pos, to_pos, matrix_shape: tuple=None, cell_size: float=None):
        if matrix_shape != None:
            assert matrix_shape[0] > 0 and matrix_shape[1] > 0
            x_step = (to_pos[0] - from_pos[0]) / matrix_shape[0]
            y_step = (to_pos[1] - from_pos[1]) / matrix_shape[1]
            result = cp.zeros(matrix_shape)
            for i in range(matrix_shape[0]):
                for j in range(matrix_shape[1]):
                    wcrd = [from_pos[0]+i*x_step, from_pos[1]+j*y_step]
                    result[i,j] = self.get_height(wcrd)

        elif cell_size != None:
            assert cell_size > 0
            mtx_shape = (int(round(abs(to_pos[0] - from_pos[0]) / cell_size, 2)), int(round(abs(to_pos[1] - from_pos[1]), 2) / cell_size))
            result = cp.zeros(mtx_shape)
            for i in range(mtx_shape[0]):
                for j in range(mtx_shape[1]):
                    wcrd = [from_pos[0]+i*cell_size, from_pos[1]+j*cell_size]
                    result[i,j] = self.get_height(wcrd)
        
        else:
            cell_size = self.cell_size
            mtx_shape = (int(abs(to_pos[0] - from_pos[0]) / cell_size), int(abs(to_pos[1] - from_pos[1]) / cell_size))
            result = cp.zeros(mtx_shape)
            for i in range(mtx_shape[0]):
                for j in range(mtx_shape[1]):
                    wcrd = [from_pos[0]+i*cell_size, from_pos[1]+j*cell_size]
                    result[i,j] = self.get_height(wcrd)
        
        return cp.asnumpy(result)



def constructHeightFieldFromImg(imgpath: str, max_height, start_pos, cell_size=0.01, data_shape=None):
    assert max_height >= 0
    img_array = mpimg.imread(imgpath)
    gray_img = cp.dot(img_array[...,:3], [0.299, 0.587, 0.114]) * max_height
    return heightField(gray_img, start_pos, cell_size=cell_size, data_shape=data_shape)


if __name__ == '__main__':
    import os
    import numpy as np
    path = os.path.abspath(os.path.dirname(__file__)) + '/pictures/test.png'
    field = constructHeightFieldFromImg(path, 3.0, [0,0], cell_size=0.03, data_shape=(60,70))
    print(field.generate_height_matrix([0.0, 0.0], [0.7, 0.5], cell_size=0.07).shape)