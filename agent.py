import numpy as np
import torch
import torch.nn.functional as F
from dataset import Dataset
import gymnasium as gym 
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
from gym_robotics_custom import HeldSetpointWrapper, VLAObservationWrapper

from torch.utils.tensorboard import SummaryWriter

class Agent:

    def __init__(self):
        max_episode_steps=500 # max episode steps
        image_size = 640 # square camera frame; downsample at train time
        env_name = "FrankaKitchen-v1"

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

        env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[task], render_mode='human')

        env = HeldSetpointWrapper(env)
        env = VLAObservationWrapper(env, image_size=image_size)



        # self.dataset = Dataset()
