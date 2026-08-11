import numpy as np

# The only seven tasks the env accepts; anything else raises at gym.make.
# "travel" is how far past the success threshold the object has to move,
# so it roughly ranks how long a demo takes.
TASKS = {
    "slide cabinet": "Slide the cabinet door open",              # travel 0.07
    "kettle":        "Move the kettle to the top left burner",   # travel 0.11
    "light switch":  "Turn on the overhead light switch",        # travel 0.39
    "microwave":     "Open the microwave door",                  # travel 0.45
    "bottom burner": "Turn the oven knob for the bottom left burner",  # 0.58
    "top burner":    "Turn the oven knob for the top left burner",     # 0.62
    "hinge cabinet": "Open the cabinet second from the left",    # travel 1.15
}

# Fixed order -- this is what task_index()'s integer ids mean. Collection
# (human_control.py) and training/eval (agent.py) both import it from here so
# they can't disagree on the mapping.
#
# V1 goal conditioning stand-in for real language conditioning: Model embeds
# this index rather than reading task_description itself. Delete this file
# and its two call sites when that lands.
TASK_DESCRIPTIONS = list(TASKS.values())
_TASK_INDEX = {description: i for i, description in enumerate(TASK_DESCRIPTIONS)}


def task_index(task_descriptions):
    """task_description string, or array of them, -> int64 array of class ids."""
    task_descriptions = np.atleast_1d(task_descriptions)
    return np.array([_TASK_INDEX[d] for d in task_descriptions], dtype=np.int64)
