import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

# agent = Agent(data_path="dataset/microwave", name="bc_microwave")
agent = Agent(data_path="dataset", name="bc_network")

agent.train(epochs=10000, batch_size=32)
