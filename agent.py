import numpy as np
import torch
import torch.nn.functional as F
import time
import datetime
import gymnasium as gym 
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
from torch.optim.adam import Adam
from gym_robotics_custom import HeldSetpointWrapper, VLAObservationWrapper, ObsReshapeWrapper 


from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset
from model import Model
from tasks import TASKS, TASK_DESCRIPTIONS, TASK_NAMES, task_index

class Agent:

    def __init__(self, eval=False, data_path="dataset", name='bc_network'):
        max_episode_steps=500
        image_size = 448
        native_image_size = 896 
        env_name = "FrankaKitchen-v1"
        max_buffer_size = 100000
        learning_rate = 0.0001

        if(eval):
            render_mode = 'human'
        else:
            render_mode = 'rgb_array'

        self.task = "microwave"
        self.task_description = TASKS[self.task]
        task_no_spaces = self.task.replace(" ", "_")

        self.env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[self.task], render_mode=render_mode)

        self.env = HeldSetpointWrapper(self.env)
        self.env = VLAObservationWrapper(self.env, image_size=native_image_size)
        self.env = ObsReshapeWrapper(self.env, image_size=image_size)

        self.dataset = Dataset(max_size=max_buffer_size, 
                               image_size=image_size, 
                               n_actions=self.env.action_space.shape[0],
                               n_joints=9)

        if not eval:
            self.dataset.load_data(path=data_path)

        obs, info = self.env.reset() 

        image_input_shape = obs['camera_scene'].shape
        joint_pos_dim     = obs['joint_pos'].shape[0]
        joint_vel_dim     = obs['joint_vel'].shape[0]
        num_actions       = self.env.action_space.shape[0] 

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.model = Model(image_input_shape=image_input_shape,
                           joint_pos_dim=joint_pos_dim,
                           joint_vel_dim=joint_vel_dim,
                           num_actions=num_actions,
                           task_dim=len(TASK_DESCRIPTIONS),
                           hidden_dim=756,
                           n_hidden_layers=2,
                           name=name).to(self.device)

        self.optimizer = Adam(self.model.parameters(), learning_rate)

        if not eval:
            self.env.close()

    def process_observation(self,obs):
        images    = obs['camera_scene']
        joint_pos = obs['joint_pos']
        joint_vel = obs['joint_vel']

        images    = torch.tensor(images, dtype=torch.float32).to(self.device) / 255
        joint_pos = torch.tensor(joint_pos, dtype=torch.float32).to(self.device)
        joint_vel = torch.tensor(joint_vel, dtype=torch.float32).to(self.device)

        if images.dim() == 3:
            images    = images.unsqueeze(0)
            joint_pos = joint_pos.unsqueeze(0)
            joint_vel = joint_vel.unsqueeze(0)

        return images, joint_pos, joint_vel



    def train(self, epochs, batch_size):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        summary_writer = SummaryWriter(summary_writer_name)

        for epoch in range(epochs):
            states, actions, _, _, tasks = self.dataset.sample_batch(batch_size)
           
            images, joint_pos, joint_vel = self.process_observation(states)

            actions = torch.tensor(actions).to(self.device)
            tasks   = torch.tensor(tasks).to(self.device)

            pred_actions = self.model(obs=images,
                                      joint_pos=joint_pos,
                                      joint_vel=joint_vel,
                                      task=tasks)            

            loss = F.mse_loss(actions, pred_actions)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            
            if(epoch % 10 == 0):
                summary_writer.add_scalar("train/loss", loss, epoch)

            if(epoch % 100 == 0):
                print(f"Epoch: {epoch} Loss: {loss.item()}")
                self.model.save_checkpoint()
                self.eval(epoch, batch_size, summary_writer)

    def eval(self, epoch, batch_size, summary_writer):
        """Validation loss per task, over every held back step."""
        self.model.eval()

        with torch.no_grad():
            for task_id in self.dataset.val_pools:
                total = 0.0
                count = 0

                for batch in self.dataset.val_batches(task_id, batch_size):
                    states, actions, _, _, tasks = batch

                    images, joint_pos, joint_vel = self.process_observation(states)
                    actions = torch.tensor(actions).to(self.device)
                    tasks   = torch.tensor(tasks).to(self.device)

                    pred_actions = self.model(obs=images,
                                              joint_pos=joint_pos,
                                              joint_vel=joint_vel,
                                              task=tasks)

                    # Summed, so a short final batch counts for what it holds.
                    total += F.mse_loss(actions, pred_actions, reduction='sum').item()
                    count += actions.numel()

                label = TASK_NAMES[task_id].replace(" ", "_")
                summary_writer.add_scalar(f"eval/{label}", total / count, epoch)
                print(f"  eval {label}: {total / count:.5f}")

        self.model.train()

    def test(self, task_description):

        self.model.load_checkpoint()

        done = False
        
        obs, info = self.env.reset()

        task_id = task_index(task_description)
        task_id = torch.tensor(task_id, dtype=torch.long).to(self.device)

        while not done:
            image, joint_pos, joint_vel = self.process_observation(obs)

            action = self.model(obs=image,
                               joint_pos=joint_pos,
                               joint_vel=joint_vel,
                               task=task_id)

            action = action.cpu().detach().numpy().squeeze()

            obs, reward, done, trunc, info = self.env.step(action)

            self.env.render()

            print(f"Obs: {obs}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"Info: {info}")

            time.sleep(0.05)
            






