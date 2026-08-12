from agent import Agent
import agent

agent = Agent()

agent.train(epochs=10000, batch_size=64)
