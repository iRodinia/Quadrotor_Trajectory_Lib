import numpy as np
from matrixGridMap import *

class GridMapPath(matrixGridMap3D):

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