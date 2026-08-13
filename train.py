from agent import Agent
import agent

# agent = Agent(data_path="dataset/microwave", name="bc_microwave")
agent = Agent(data_path="dataset", name="bc_network")

agent.train(epochs=10000, batch_size=32)
