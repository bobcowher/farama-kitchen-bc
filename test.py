from agent import Agent
import agent

from tasks import TASKS


task = "microwave"
# task = "hinge cabinet"
task_description = TASKS[task]

agent = Agent(eval=True)

agent.test(task_description=task_description)
