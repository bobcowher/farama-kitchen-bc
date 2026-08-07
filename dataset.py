import numpy as np
import os
from datetime import datetime

class DatasetShard():
    """One collection run's worth of VLA transitions, saved as its own file.

    Appends rather than preallocating. The other buffers in this repo size a
    uint8 arena up front, which works for small frames; two 640x640 views can't
    be sized that way (replay_buffer_size would ask for petabytes).

    Frames are stored as raw uint8 RGB and zlib'd by savez_compressed, ~450 KiB
    per timestep for both views at 640.

    next_camera_* is not stored. It is the next step's frame by construction, so
    keeping it would double the file for nothing; rebuild it at load time.
    """

    def __init__(self, task_name, task_description):
        self.mem_ctr = 0
        self.task_name = task_name
        self.task_description = task_description
        self.camera_scene_memory = []
        self.camera_wrist_memory = []
        self.joint_pos_memory = []
        self.joint_vel_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.new_joint_pos_memory = []
        self.new_joint_vel_memory = []
        self.terminal_memory = []

    def __len__(self):
        return self.mem_ctr

    def store_transition(self, state, action, reward, state_, done):
        self.camera_scene_memory.append(state["camera_scene"])
        self.camera_wrist_memory.append(state["camera_wrist"])
        self.joint_pos_memory.append(state["joint_pos"])
        self.joint_vel_memory.append(state["joint_vel"])
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.new_joint_pos_memory.append(state_["joint_pos"])
        self.new_joint_vel_memory.append(state_["joint_vel"])
        self.terminal_memory.append(done)

        self.mem_ctr += 1

    def save_to_csv(self):

        date_formatted = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        directory = f'dataset/{self.task_name}'
        filename = f'{directory}/shard_{date_formatted}.npz'

        os.makedirs(directory, exist_ok=True)

        np.savez_compressed(filename,
                 task_name=self.task_name,
                 task_description=self.task_description,
                 camera_scene=np.stack(self.camera_scene_memory),
                 camera_wrist=np.stack(self.camera_wrist_memory),
                 joint_pos=np.stack(self.joint_pos_memory),
                 joint_vel=np.stack(self.joint_vel_memory),
                 action=np.stack(self.action_memory),
                 reward=np.array(self.reward_memory),
                 next_joint_pos=np.stack(self.new_joint_pos_memory),
                 next_joint_vel=np.stack(self.new_joint_vel_memory),
                 done=np.array(self.terminal_memory, dtype=bool))
        print("-" * 20)
        print(f"Saved {filename} ({self.mem_ctr} steps, "
              f"{os.path.getsize(filename) / 1e6:.1f} MB)")
        print("-" * 20)
