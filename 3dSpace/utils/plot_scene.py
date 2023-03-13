import numpy as np
import matplotlib.pyplot as plt
from matrixGridmap import *

def plot_girdMap(girdMap: OccupancyGridMap3D):
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

def plot_shape(trajectory, dt=0.01):
    """draw 4 subplots:
        1. 3D trajectory
        2. x-axis reference signal
        3. y-axis reference signal
        4. z-axis reference signal

    Args:
        trajectory: the trajectory object
        dt: discretize accuracy
    """
    timestamps = np.arange(0, trajectory.duration+dt, dt)
    key_points = np.array([trajectory.eval(t) for t in timestamps])[:,:3]

    fig_tracking = plt.figure("Trajectory shape", figsize=(13, 3.2))
    ax1 = fig_tracking.add_subplot(141, projection='3d')
    ax2 = fig_tracking.add_subplot(142)
    ax3 = fig_tracking.add_subplot(143)
    ax4 = fig_tracking.add_subplot(144)

    ax1.plot(key_points[:,0], key_points[:,1], key_points[:,2], color='black')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    ax2.plot(timestamps, key_points[:,0], color='red')
    ax2.set_xlabel('time /s')
    ax2.set_ylabel('x /m')

    ax3.plot(timestamps, key_points[:,1], color='green')
    ax3.set_xlabel('time /s')
    ax3.set_ylabel('y /m')

    ax4.plot(timestamps, key_points[:,2], color='blue')
    ax4.set_xlabel('time /s')
    ax4.set_ylabel('z /m')

    fig_tracking.subplots_adjust(
        left=0,
        bottom=0.143,
        right=0.971,
        top=0.948,
        wspace=0.224,
        hspace=0.2
    )


def plot_curv_and_tors(trajectory: FrenetTraj, dt=0.01):
    """draw 2 subplots:
        1. curvature
        2. torsion

    Args:
        trajectory: the trajectory object
        dt: discretize accuracy
    """
    timestamps = np.arange(0, trajectory.duration, dt)
    key_curv = np.array([trajectory.eval_curvature(t) for t in timestamps])
    key_tors = np.array([trajectory.eval_torsion(t) for t in timestamps])

    fig_tracking = plt.figure("Curvature and torsion", figsize=(6.5, 3.2))
    ax1 = fig_tracking.add_subplot(121)
    ax2 = fig_tracking.add_subplot(122)

    ax1.plot(timestamps, key_curv, color='orangered')
    ax2.plot(timestamps, key_tors, color='fuchsia')

    fig_tracking.subplots_adjust(
        left=0,
        bottom=0.143,
        right=0.971,
        top=0.948,
        wspace=0.224,
        hspace=0.2
    )