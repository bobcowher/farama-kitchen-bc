import numpy as np
import os
from datetime import datetime

class DatasetShard():
    def __init__(self, max_size, input_size, n_actions, task_name, task_description, 
                 augment_data=False, augment_rewards=False, expert_data_ratio=0.1,
                 augment_noise_ratio=0.1):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, input_size))
        self.new_state_memory = np.zeros((self.mem_size, input_size))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.task_description = task_description
        self.task_name = task_name




    def __len__(self):
        return self.mem_ctr

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def save_to_csv(self):
        
        date_formatted = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        directory = f'dataset/{self.task_name}'
        filename = f'{directory}/shard_{date_formatted}.npz'

        os.makedirs(directory, exist_ok=True)

        np.savez(filename,
                 task_description=self.task_description,
                 state=self.state_memory[:self.mem_ctr],
                 action=self.action_memory[:self.mem_ctr],
                 reward=self.reward_memory[:self.mem_ctr],
                 next_state=self.new_state_memory[:self.mem_ctr],
                 done=self.terminal_memory[:self.mem_ctr])
        print("-" * 20)
        print(f"Saved {filename}")
        print("-" * 20)

