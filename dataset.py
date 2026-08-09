import numpy as np
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import glob
import frames

# Loading is 77% zlib inflate and 23% resize, both of which release the GIL, so
# threads scale here where they normally would not: 8x on 32 cores, flat past 16
# workers because the limit is memory bandwidth rather than CPU.
WORKERS = min(16, os.cpu_count() or 1)


def _open(filename):
    """np.load, but say which shard is bad rather than raising from zipfile."""
    try:
        return np.load(filename)
    except Exception as e:
        raise ValueError(
            f"{filename} could not be read ({type(e).__name__}). A shard left "
            f"half-written by a killed collector looks like this; delete it.") from e


class Dataset():
    """Demonstrations loaded off disk and sampled in batches for behavior cloning.

    Collection is DatasetShard's job; this only reads the shards it wrote.

    The arenas mirror the observation space in gym_robotics_custom.py, and
    sample_batch hands back a dict with those same keys, so a policy written
    against the live env works unchanged on the dataset. camera_scene comes
    back as NCHW to match Conv2d, the same as ObsReshapeWrapper gives at
    rollout; the arena itself stays HWC, which is what the shards hold and what
    zlib compresses well.

    max_size is a declaration of what you expect to load. load_data refuses to
    exceed it rather than quietly growing, so a directory holding more than you
    planned for is an error you hear about instead of a machine that swaps.

    Shards are archived at whatever the collector rendered, currently 896, and
    load_data reduces them to image_size on the way in. image_size must divide
    the shard width evenly; frames.py explains why. Declaring a smaller arena
    is the lever that keeps a large dataset in memory: at 896 a step costs
    2.30 MiB, at 224 it costs 147 KiB, at 128 it costs 48 KiB.

    Load high enough for the encoder you might want later -- 224 is what OpenVLA,
    pi0 and PaliGemma all take -- and take any further step down per batch.
    Reducing here is one way; the frames never go back up.
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
        steps_per_file = [len(_open(filename)['action']) for filename in files]
        total = sum(steps_per_file)

        if total > self.mem_size:
            raise ValueError(
                f"{path} holds {total} steps across {len(files)} shards, but this "
                f"Dataset was built for {self.mem_size}. Raise max_size to at least "
                f"{total} ({total * self.camera_scene_memory[0].nbytes / 1e9:.1f} GB "
                f"of frames) or point at a smaller directory.")

        index = 0
        # map keeps input order, which is what lets the zip below line results
        # up with the counts from the first pass.
        with ThreadPoolExecutor(WORKERS) as pool:
            loaded = pool.map(self._read, files)

            for filename, steps, (camera, data) in zip(files, steps_per_file, loaded):
                print(f"  {filename}: {steps} steps")

                end = index + steps
                self.camera_scene_memory[index:end] = camera
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

    def _read(self, filename):
        """Decompress one shard and reduce its frames. Runs on a worker thread.

        The resize happens here rather than in the caller so the expensive half
        is parallel too, and so full-size frames are freed as soon as the
        reduced copy exists.
        """
        data = _open(filename)
        try:
            camera = frames.resize(data['camera_scene'],
                                   self.camera_scene_memory.shape[1])
        except ValueError as e:
            # resize only sees an array, so it cannot say which of a few
            # hundred shards is the odd one out.
            raise ValueError(f"{filename}: {e}") from None
        return camera, data

    def sample_batch(self, batch_size):
        batch = np.random.choice(self.mem_ctr, batch_size)

        state = {
            # NCHW, because that is the shape Conv2d takes. Fancy indexing has
            # already made a fresh contiguous HWC copy, so the transpose is a
            # view over it -- nothing moves, and torch.from_numpy reports the
            # result channels_last-contiguous, which is the fast layout on GPU.
            # Storage stays HWC; see the class docstring.
            "camera_scene": self.camera_scene_memory[batch].transpose(0, 3, 1, 2),
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

    Frames go in HWC, straight off the renderer. Do not record through
    ObsReshapeWrapper -- it hands out NCHW for the model, and the arena would
    reject the shape.

    Frames are stored as raw uint8 RGB and zlib'd by savez_compressed. Measured
    over 104 microwave demos: 509 KiB per timestep, 59 steps per demo, 3.2 GB.
    Demo length is what drives the total, not the task count -- microwave is a
    short one, so scale by the task before trusting an estimate. The arena
    itself is image_size^2 * 3 * max_size, 1.20 GB at 896x896x500.

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

        # Filenames are second-resolution, so two saves inside one second would
        # otherwise overwrite silently. A demo takes far longer than that to
        # record, so a collision means something is wrong, not that we are fast.
        if os.path.exists(filename):
            raise FileExistsError(
                f"{filename} already exists. Shard names are second-resolution, "
                f"so this shard would overwrite one saved in the same second.")

        # Written to a temp name and moved into place, because savez_compressed
        # takes about a second on a 25 MB shard and the file is an invalid zip
        # for all of it. os.replace is atomic, so a reader either sees the whole
        # shard or no file at all -- which makes it safe to load a directory
        # while a collection run is still appending to it. A killed collector
        # leaves a visible .tmp rather than a corrupt shard that every future
        # load_data would die on. load_data globs *.npz, so strays are ignored.
        tmp = f'{filename}.tmp'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh,
                     task_name=self.task_name,
                     task_description=self.task_description,
                     camera_scene=self.camera_scene_memory[:self.mem_ctr],
                     joint_pos=self.joint_pos_memory[:self.mem_ctr],
                     joint_vel=self.joint_vel_memory[:self.mem_ctr],
                     action=self.action_memory[:self.mem_ctr],
                     reward=self.reward_memory[:self.mem_ctr],
                     done=self.terminal_memory[:self.mem_ctr])
        os.replace(tmp, filename)

        print("-" * 20)
        print(f"Saved {filename} ({self.mem_ctr} steps, "
              f"{os.path.getsize(filename) / 1e6:.1f} MB)")
        print("-" * 20)
