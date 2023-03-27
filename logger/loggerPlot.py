import numpy as np
import matplotlib.pyplot as plt
from .quadrotorLogger import quadrotorLogger


def quad_log_plot(logger: quadrotorLogger, title: str="Quadrotor Simulation Result"):
    """draw 4 subplots:
        1. 3D reference trajectory *2 and real trajectory *2
        2. x-axis original reference, direct tracking positions, resampled reference and resampled tracking positions
        3. y-axis original reference, direct tracking positions, resampled reference and resampled tracking positions
        4. z-axis original reference, direct tracking positions, resampled reference and resampled tracking positions

        original reference: blue
        tracking states: red
        controls: green
    Args:
        TBD
    """

    timestamps = logger.timestamps
    states = logger.states
    targets = logger.targets
    controls = logger.controls

    fig_tracking = plt.figure(title, figsize=(13, 3.2))
    ax1 = fig_tracking.add_subplot(141, projection='3d')
    ax2 = fig_tracking.add_subplot(142)
    ax3 = fig_tracking.add_subplot(143)
    ax4 = fig_tracking.add_subplot(144)

    ax1.plot(targets[:,0], targets[:,1], targets[:,2], color='blue')
    ax1.plot(states[:,0], states[:,1], states[:,2], color='red')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    ax2.plot(timestamps, targets[:,0], color='blue')
    ax2.plot(timestamps, states[:,0], color='red')
    ax2.set_xlabel('time /s')
    ax2.set_ylabel('x /m')

    ax3.plot(timestamps, targets[:,1], color='blue')
    ax3.plot(timestamps, states[:,1], color='red')
    ax3.set_xlabel('time /s')
    ax3.set_ylabel('y /m')

    ax4.plot(timestamps, targets[:,2], color='blue')
    ax4.plot(timestamps, states[:,2], color='red')
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
