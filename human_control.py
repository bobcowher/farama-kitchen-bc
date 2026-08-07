import time
import os
import gymnasium as gym
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
import numpy as np
from dataset import DatasetShard
import datetime
# from agent import Agent
from gym_robotics_custom import HeldSetpointWrapper, VLAObservationWrapper
from torch.utils.tensorboard import SummaryWriter
import pygame
from controller import Controller


if __name__ == '__main__':

    replay_buffer_size = 1000
    episodes = 10
    warmup = 20
    batch_size = 64
    updates_per_step = 1
    gamma = 0.99
    tau = 0.005
    alpha = 0.15 # Temperature parameter.
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = False
    hidden_size = 512
    learning_rate = 0.0001
    max_episode_steps=500 # max episode steps
    image_size = 640 # square camera frame; downsample at train time
    env_name = "FrankaKitchen-v1"
    exploration_scaling_factor=0.01

    task = "hinge cabinet"
    task_description = "Open the cabinet second from the left"

    # The only seven tasks the env accepts; anything else raises at gym.make.
    # Swap both lines together. Descriptions are placeholders in the env's own
    # terms -- reword them to match how you'd phrase the instruction.
    # "travel" is how far past the success threshold the object has to move,
    # so it roughly ranks how long a demo takes.
    #
    # task = "slide cabinet"
    # task_description = "Slide the cabinet door open"          # travel 0.07
    #
    # task = "kettle"
    # task_description = "Move the kettle to the top left burner"  # travel 0.11
    #
    # task = "light switch"
    # task_description = "Turn on the overhead light switch"    # travel 0.39
    #
    # task = "microwave"
    # task_description = "Open the microwave door"              # travel 0.45
    #
    # task = "bottom burner"
    # task_description = "Turn the oven knob for the bottom left burner"  # 0.58
    #
    # task = "top burner"
    # task_description = "Turn the oven knob for the top left burner"     # 0.62
    #
    # task = "hinge cabinet"
    # task_description = "Open the cabinet second from the left"  # travel 1.15

    task_no_spaces = task.replace(" ", "_")


    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[task], render_mode='human')

    env = HeldSetpointWrapper(env)
    env = VLAObservationWrapper(env, image_size=image_size)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    controller = Controller()

    while True: # Run until interrupted
        episode_steps = 0
        done = False
        state, info = env.reset()

        memory = DatasetShard(max_size=max_episode_steps,
                              image_size=image_size,
                              n_actions=env.action_space.shape[0],
                              task_name=task_no_spaces,
                              task_description=task_description)
        starting_memory_size = 0 # TODO: Come back and make this dynamic. We have the technology, we can rebuild the data

        reward = 0

        while not done and episode_steps < max_episode_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                # Check if CTRL+H is pressed
                    if event.key == pygame.K_h and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        # Trigger the key event in MuJoCo
                        env.render()  # Ensure the environment handles the key event

            action = controller.get_action()
            if(action is not None):
                next_state, reward, done, _, _ = env.step(action)
                memory.store_transition(state, action, reward, done)
                print(f"Episode step: {episode_steps} Reward: , {reward} Successfully added {memory.mem_ctr - starting_memory_size} steps to memory. Total: {memory.mem_ctr}")
                state = next_state
                episode_steps += 1
            time.sleep(0.05)
        

        if reward > 0:
            memory.save_to_csv()



