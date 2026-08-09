import numpy as np
import torch
import torch.nn.functional as F
from dataset import Dataset
import gymnasium as gym 
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
from gym_robotics_custom import HeldSetpointWrapper, VLAObservationWrapper, ObsReshapeWrapper 

from torch.utils.tensorboard import SummaryWriter

class Agent:

    def __init__(self):
        max_episode_steps=500 # max episode steps
        image_size = 448 
        native_image_size = 896 
        env_name = "FrankaKitchen-v1"
        max_buffer_size = 100000

        # The only seven tasks the env accepts; anything else raises at gym.make.
        # "travel" is how far past the success threshold the object has to move,
        # so it roughly ranks how long a demo takes.
        TASKS = {
            "slide cabinet": "Slide the cabinet door open",              # travel 0.07
            "kettle":        "Move the kettle to the top left burner",   # travel 0.11
            "light switch":  "Turn on the overhead light switch",        # travel 0.39
            "microwave":     "Open the microwave door",                  # travel 0.45
            "bottom burner": "Turn the oven knob for the bottom left burner",  # 0.58
            "top burner":    "Turn the oven knob for the top left burner",     # 0.62
            "hinge cabinet": "Open the cabinet second from the left",    # travel 1.15
        }

        task = "microwave"
        task_description = TASKS[task]
        task_no_spaces = task.replace(" ", "_")

        self.env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[task], render_mode='human')

        self.env = HeldSetpointWrapper(self.env)
        self.env = VLAObservationWrapper(self.env, image_size=native_image_size)
        self.env = ObsReshapeWrapper(self.env, image_size=image_size)

        self.dataset = Dataset(max_size=max_buffer_size, 
                               image_size=image_size, 
                               n_actions=self.env.action_space.shape[0],
                               n_joints=9)

        self.dataset.load_data()
