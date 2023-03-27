import os
import numpy as np
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class quadrotorLogger(object):
    """A class for logging quadrotor simulation data.

    Stores, saves to file, and plots the kinematic information

    """
    def __init__(self, output_folder: str="results", max_log_entries: int=10000):
        """Logger class __init__ method.

        output_folder : str, optional
            Used to determine the output folder name. Can be absolute path

        """
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        self.valid_entry_num = 0
        self.max_log_num = max_log_entries

        self.timestamps = np.zeros(max_log_entries)
        # store timestamps
        self.states = np.zeros((self.max_log_num, 12))
        # 12 states: 
        # (0)pos_x, (1)pos_y, (2)pos_z,
        # (3)vel_x, (4)vel_y, (5)vel_z,
        # (6)roll, (7)pitch, (8)yaw,
        # (9)ang_vel_x, (10)ang_vel_y, (11)ang_vel_z
        self.targets = np.zeros((self.max_log_num, 13))
        # 13 targets: 
        # (0)pos_x, (1)pos_y, (2)pos_z,
        # (3)vel_x, (4)vel_y, (5)vel_z,
        # (6)acc_x, (7)acc_y, (8)acc_z,
        # (9)ang_vel_x, (10)ang_vel_y, (11)ang_vel_z,
        # (12)yaw
        self.controls = np.zeros((self.max_log_num, 4))
        # 4 control inputs (Force and Moment):
        # (0)thrust, (1)moment_x, (2)moment_y, (3)moment_z
        self.data_dict = {}
        # Other data to be logged
        # In the form: "data_name": data_array

    def ctrl_step_log(self, timestamp, state, target, control):
        """Logs entries for a single simulation step.

        Parameters
        ----------
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (12,)-shaped array of floats containing the drone's state.
        target : ndarray
            (13,)-shaped array of floats containing the drone's reference target.
        control : ndarray
            (4,)-shaped array of floats containing the drone's control input.
        """
        if self.valid_entry_num >= self.max_log_num:
            print("Maximum log entries reached! Log nothing!")
            return
        
        self.timestamps[self.valid_entry_num] = timestamp
        self.states[self.valid_entry_num, :] = state
        self.targets[self.valid_entry_num, :] = target
        self.controls[self.valid_entry_num, :] = control

        self.valid_entry_num += 1


    def extend_log(self, data_name: str, data, sequential=False):
        if data_name in self.data_dict:
            if sequential:
                self.data_dict[data_name].append(data)
            else:
                self.data_dict[data_name] = data
        else:
            if sequential:
                self.data_dict[data_name] = [data]
            else:
                self.data_dict[data_name] = data


    def save(self, filename: str=None, front_title=""):
        """Save the logs to file.
        """
        timestamps = self.timestamps[:self.valid_entry_num]
        states = self.states[:self.valid_entry_num, :]
        targets = self.targets[:self.valid_entry_num, :]
        controls = self.controls[:self.valid_entry_num, :]

        if filename is not None:
             with open(os.path.join(self.output_folder, filename+".npy"), 'wb') as out_file:
                np.savez(
                    out_file,
                    timestamps=timestamps,
                    states=states,
                    targets=targets,
                    controls=controls,
                    other_data=self.data_dict
                )
        else:
            with open(os.path.join(self.output_folder, front_title+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".npy"), 'wb') as out_file:
                np.savez(
                    out_file,
                    timestamps=timestamps,
                    states=states,
                    targets=targets,
                    controls=controls,
                    other_data=self.data_dict
                )