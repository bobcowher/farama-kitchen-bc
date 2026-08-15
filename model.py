import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, image_input_shape, 
                 joint_pos_dim, 
                 joint_vel_dim, 
                 task_dim,
                 num_actions, 
                 hidden_dim,
                 compression_dim=None,
                 n_hidden_layers=1,
                 use_layer_norm=False,
                 checkpoint_dir='checkpoints',
                 name='bc_network'):
        super(Model, self).__init__()

        # Per-modality embedding width before fusion. Defaults to hidden_dim
        # (the historical behavior); pass a value to tune it independently.
        if compression_dim is None:
            compression_dim = hidden_dim

        self.conv1 = nn.Conv2d(image_input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, *image_input_shape)
            flat_size = self._conv_forward(dummy).shape[1]

        # We want the total compression dim to be the joint information, for gut feeling reasons. 
        compression_dim_small = compression_dim // 2

        self.joint_pos_input = nn.Linear(joint_pos_dim, compression_dim_small)
        self.joint_vel_input = nn.Linear(joint_vel_dim, compression_dim_small)
        self.task_input      = nn.Embedding(task_dim, compression_dim_small)


        self.image_input = nn.Linear(flat_size, compression_dim)

        self.compression_layer = nn.Linear(compression_dim + (compression_dim_small * 3), hidden_dim)

        # n_hidden_layers hidden FC layers between fusion and output.
        # n_hidden_layers=1 reproduces the original single `linear1`.
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )

        # Optional per-layer LayerNorm, applied Linear -> LayerNorm -> ReLU.
        # Off by default so existing checkpoints keep loading unchanged.
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.hidden_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(n_hidden_layers)]
            )

        self.output = nn.Linear(hidden_dim, num_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.apply(weights_init_)


    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)

    def forward(self, obs, joint_pos, joint_vel, task):
        x_image = self._conv_forward(obs)
        x_image = F.relu(self.image_input(x_image))

        x_joint_pos = F.relu(self.joint_pos_input(joint_pos))
        x_joint_vel = F.relu(self.joint_vel_input(joint_vel))
        x_task      = F.relu(self.task_input(task))

        # x_task = self.task_input(task_id)

        x = torch.cat([x_image, x_joint_pos, x_joint_vel, x_task], dim=1)

        x = F.relu(self.compression_layer(x))

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.hidden_norms[i](x)
            x = F.relu(x)

        x = F.tanh(self.output(x))
        # # x = F.tanh(self.out)
        # x = self.output(x)
        # # x = F.tanh(x)
        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

