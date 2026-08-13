from agent import Agent

agent = Agent(eval=True)
agent.model.load_checkpoint()

agent.test("microwave", render_mode="human", delay=0.05)
