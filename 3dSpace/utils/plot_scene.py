import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from vispy import app, visuals, scene
from matrixGridMap import *

def plot_girdMap(gridMap: matrixGridMap3D):
    """
    plot the grid map
    """
    vol = gridMap.get_data().astype(np.int16) * 100
    vol = vol.transpose((2,1,0))

    canvas = scene.SceneCanvas(keys='interactive', title='Grid Map 3D', show=True)
    view = canvas.central_widget.add_view()
    volume = scene.visuals.Volume(vol, clim=(0,200), parent=view.scene, method='mip')
    cam = scene.cameras.TurntableCamera(parent=view.scene, name='Turntable')
    view.camera = cam

    canvas.app.run()

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


def plot_curv_and_tors(trajectory, dt=0.01):
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

if __name__ == "__main__":
    import os
    import time
    from heightField import constructHeightFieldFromImg

    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/pictures/test.png'
    field = constructHeightFieldFromImg(path, 3.0, [0,0], cell_size=0.03, data_shape=(60,70))
    obs_dict = {'obs1': {'type': 'height_field', 'height_field':field}}

    start_time = time.time()
    map = matrixGridMap3D(center_crd=[0,0,0], half_extend=[3,3,2], cell_size=0.05)
    cons_time = time.time()
    print('construction time: ', cons_time - start_time)
    map.add_obstacles(obs_dict=obs_dict)
    load_time = time.time()
    print('load time: ', load_time - cons_time)

    plot_girdMap(map)