import numpy as np
import os
from datetime import datetime
import glob

class Dataset():
    """Demonstrations loaded off disk and sampled in batches for behavior cloning.

    Collection is DatasetShard's job; this only reads the shards it wrote.

    The arenas mirror the observation space in gym_robotics_custom.py, and
    sample_batch hands back a dict with those same keys, so a policy written
    against the live env works unchanged on the dataset.

    max_size is a declaration of what you expect to load. load_data refuses to
    exceed it rather than quietly growing, so a directory holding more than you
    planned for is an error you hear about instead of a machine that swaps.

    Frames cost image_size^2 * 3 bytes per step -- 1.17 MiB at 640 -- so a 91 GB
    machine holds roughly one task's worth of demos. Past that this needs to
    become a lazy loader that pulls frames per batch.
    """

    def __init__(self, max_size, image_size, n_actions, n_joints=9):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.camera_scene_memory = np.zeros(
            (self.mem_size, image_size, image_size, 3), dtype=np.uint8)
        self.joint_pos_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.joint_vel_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.task_description_memory = np.zeros(self.mem_size, dtype=object)

    def __len__(self):
        return self.mem_ctr

    def can_sample(self, batch_size):
        return self.mem_ctr >= batch_size

    def load_data(self, path):
        """Load every npz shard under path, recursively.

        Shards are concatenated in sorted filename order, so a directory of
        tasks loads as one mixed set. That is the normal case now that the
        policy is language conditioned on task_description.
        """
        files = sorted(glob.glob(os.path.join(path, '**', '*.npz'), recursive=True))

        if not files:
            print(f"No npz files found under {path}")
            return

        # First pass counts, so an oversized directory fails before anything is
        # written. Only the small action array is read here; the frames are read
        # once, in the fill pass below.
        steps_per_file = [len(np.load(filename)['action']) for filename in files]
        total = sum(steps_per_file)

        if total > self.mem_size:
            raise ValueError(
                f"{path} holds {total} steps across {len(files)} shards, but this "
                f"Dataset was built for {self.mem_size}. Raise max_size to at least "
                f"{total} ({total * self.camera_scene_memory[0].nbytes / 1e9:.1f} GB "
                f"of frames) or point at a smaller directory.")

        index = 0
        for filename, steps in zip(files, steps_per_file):
            data = np.load(filename)

            print(f"  {filename}: {steps} steps")

            end = index + steps
            self.camera_scene_memory[index:end] = data['camera_scene']
            self.joint_pos_memory[index:end] = data['joint_pos']
            self.joint_vel_memory[index:end] = data['joint_vel']
            self.action_memory[index:end] = data['action']
            self.reward_memory[index:end] = data['reward']
            self.terminal_memory[index:end] = data['done']
            self.task_description_memory[index:end] = str(data['task_description'])
            index = end

        self.mem_ctr = total

        print(f"Loaded {len(files)} shards, {self.mem_ctr} of {self.mem_size} steps, "
              f"{self.camera_scene_memory[:self.mem_ctr].nbytes / 1e9:.1f} GB of frames")

    def sample_batch(self, batch_size):
        batch = np.random.choice(self.mem_ctr, batch_size)

        state = {
            "camera_scene": self.camera_scene_memory[batch],
            "joint_pos": self.joint_pos_memory[batch],
            "joint_vel": self.joint_vel_memory[batch],
        }

        return (state,
                self.action_memory[batch],
                self.reward_memory[batch],
                self.terminal_memory[batch],
                self.task_description_memory[batch])


class DatasetShard():
    """One collection run's worth of VLA transitions, saved as its own file.

    max_size is a declaration of how long an episode you expect, normally
    max_episode_steps. Overrunning it raises rather than wrapping, because a
    shard is a single episode and a ring buffer would corrupt it instead of
    ageing out old data.

    Frames are stored as raw uint8 RGB and zlib'd by savez_compressed, ~280 KiB
    per timestep at 640, so roughly 145 GB for 300 demos across 7 tasks. The
    arena itself is image_size^2 * 3 * max_size, 614 MB at 640x640x500.

    No next_state is stored, in any form. Behavior cloning never reads it, and
    within a shard it is an exact duplicate of the following row.
    """

    def __init__(self, max_size, image_size, n_actions, task_name,
                 task_description, n_joints=9):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.task_name = task_name
        self.task_description = task_description
        self.camera_scene_memory = np.zeros(
            (self.mem_size, image_size, image_size, 3), dtype=np.uint8)
        self.joint_pos_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.joint_vel_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def __len__(self):
        return self.mem_ctr

    def store_transition(self, state, action, reward, done):
        if self.mem_ctr >= self.mem_size:
            raise ValueError(
                f"shard is full at {self.mem_size} steps. Raise max_size to match "
                f"max_episode_steps, or the episode is running longer than expected.")

        index = self.mem_ctr

        self.camera_scene_memory[index] = state["camera_scene"]
        self.joint_pos_memory[index] = state["joint_pos"]
        self.joint_vel_memory[index] = state["joint_vel"]
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def save_to_csv(self):

        date_formatted = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        directory = f'dataset/{self.task_name}'
        filename = f'{directory}/shard_{date_formatted}.npz'

        os.makedirs(directory, exist_ok=True)

        np.savez_compressed(filename,
                 task_name=self.task_name,
                 task_description=self.task_description,
                 camera_scene=self.camera_scene_memory[:self.mem_ctr],
                 joint_pos=self.joint_pos_memory[:self.mem_ctr],
                 joint_vel=self.joint_vel_memory[:self.mem_ctr],
                 action=self.action_memory[:self.mem_ctr],
                 reward=self.reward_memory[:self.mem_ctr],
                 done=self.terminal_memory[:self.mem_ctr])
        print("-" * 20)
        print(f"Saved {filename} ({self.mem_ctr} steps, "
              f"{os.path.getsize(filename) / 1e6:.1f} MB)")
        print("-" * 20)
