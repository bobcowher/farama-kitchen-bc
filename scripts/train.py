import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

# Model-size/depth sweep knobs. Each tuning-h<dim>-l<layers> branch overrides
# just these two lines; everything else is shared sweep infrastructure.
HIDDEN_DIM = 756
N_HIDDEN_LAYERS = 2

epochs = 10001
# epochs = 100001

agent = Agent(data_path="dataset", name=f"bc_h{HIDDEN_DIM}_l{N_HIDDEN_LAYERS}",
              hidden_dim=HIDDEN_DIM, n_hidden_layers=N_HIDDEN_LAYERS)

agent.train(epochs=epochs, batch_size=64)
