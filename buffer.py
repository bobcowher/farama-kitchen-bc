import numpy as np
import os

class ReplayBuffer():
    def __init__(self, max_size, input_size, n_actions, sad_robot=False, 
                 augment_data=False, augment_rewards=False, expert_data_ratio=0.1,
                 augment_noise_ratio=0.1):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, input_size))
        self.new_state_memory = np.zeros((self.mem_size, input_size))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.sad_robot = sad_robot
        self.augment_data = augment_data
        self.augment_rewards = augment_rewards
        self.augment_noise_ratio = augment_noise_ratio # Only relevant if augment rewards is set. 
        self.expert_data_ratio = expert_data_ratio
        self.expert_data_cutoff = 0


    def __len__(self):
        return self.mem_ctr

    def can_sample(self, batch_size):
        if self.mem_ctr > (batch_size * 1000):
            return True
        else:
            return False

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
        
        if self.expert_data_ratio > 0:
            expert_batch_quantity = int(batch_size * self.expert_data_ratio)
            random_batch = np.random.choice(max_mem, batch_size - expert_batch_quantity)
            expert_batch = np.random.choice(self.expert_data_cutoff, expert_batch_quantity)
            batch = np.concatenate((random_batch, expert_batch))
        else:
            batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        if self.augment_data:
            # Compute dynamic noise levels based on the average absolute values
            state_noise_std = self.augment_noise_ratio * np.mean(np.abs(states))
            action_noise_std = self.augment_noise_ratio * np.mean(np.abs(actions))
            reward_noise_std = self.augment_noise_ratio * np.mean(np.abs(rewards))

            # Adding dynamic noise to states, actions, and rewards
            states = states + np.random.normal(0, state_noise_std, states.shape)
            actions = actions + np.random.normal(0, action_noise_std, actions.shape)
            # rewards = rewards + np.random.normal(0, reward_noise_std, rewards.shape)

        if self.augment_rewards:
            rewards = rewards * 100

            if self.sad_robot:
                rewards = rewards - 1

        return states, actions, rewards, states_, dones

    def save_to_csv(self, filename='checkpoints/memory.npz'):

        os.makedirs('checkpoints', exist_ok=True)

        np.savez(filename,
                 state=self.state_memory[:self.mem_ctr],
                 action=self.action_memory[:self.mem_ctr],
                 reward=self.reward_memory[:self.mem_ctr],
                 next_state=self.new_state_memory[:self.mem_ctr],
                 done=self.terminal_memory[:self.mem_ctr])
        print(f"Saved {filename}")

    def load_from_csv(self, filename='checkpoints/memory.npz', expert_data=True):
        try:
            data = np.load(filename)
            self.mem_ctr = len(data['state'])
            self.state_memory[:self.mem_ctr] = data['state']
            self.action_memory[:self.mem_ctr] = data['action']
            self.reward_memory[:self.mem_ctr] = data['reward']
            self.new_state_memory[:self.mem_ctr] = data['next_state']
            self.terminal_memory[:self.mem_ctr] = data['done']
            print(f"Successfully loaded {filename} into memory")
            print(f"{self.mem_ctr} memories loaded")
            
            if(expert_data):
                self.expert_data = expert_data
                self.expert_data_cutoff = self.mem_ctr

        except:
            print(f"Unable to load memory from ")


class CombinedReplayBuffer:

    def __init__(self, buffers, percentages):
        if len(buffers) != len(percentages):
            raise ValueError("Number of buffers must match number of percentages")

        if not np.isclose(sum(percentages), 1.0):
            raise ValueError("Percentages must sum to 1")

        self.buffers = buffers
        self.percentages = percentages
        
    def sample_buffer(self, batch_size):
        sizes = [int(batch_size * perc) for perc in self.percentages]
        
        if any(not buf.can_sample(size) for buf, size in zip(self.buffers, sizes)):
            raise ValueError("One of the buffers cannot currently sample the required batch size")
        
        sampled_data = [buf.sample_buffer(size) for buf, size in zip(self.buffers, sizes)]
        
        states, actions, rewards, states_, dones = zip(*sampled_data)
        
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        states_ = np.concatenate(states_, axis=0)
        dones = np.concatenate(dones, axis=0)
        
        return states, actions, rewards, states_, dones
