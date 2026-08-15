import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

# Model-size/depth sweep knobs. Each tuning-h<dim>-l<layers>[-ln] branch
# overrides these lines; everything else is shared sweep infrastructure.
HIDDEN_DIM = 756
N_HIDDEN_LAYERS = 3
USE_LAYER_NORM = True

epochs = 10001
# epochs = 100001

ln_suffix = "_ln" if USE_LAYER_NORM else ""
agent = Agent(data_path="dataset", name=f"bc_h{HIDDEN_DIM}_l{N_HIDDEN_LAYERS}{ln_suffix}",
              hidden_dim=HIDDEN_DIM, n_hidden_layers=N_HIDDEN_LAYERS,
              use_layer_norm=USE_LAYER_NORM)

agent.train(epochs=epochs, batch_size=64)
