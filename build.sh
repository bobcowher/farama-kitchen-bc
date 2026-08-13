#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate farama-kitchen-bc

# Run once
# python $(python -c "import robosuite, os; print(os.path.dirname(robosuite.__file__))")/scripts/setup_macros.py
# export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning"
# export ROBOSUITE_LOG_LEVEL=ERROR   # robosuite uses its own logger, not warnings

#pip install -r requirements.txt

# The main script: training via scripts/train.py. Watch one rollout with
# ./scripts/test.py, collect demos with ./scripts/human_control.py.
# python -u ./scripts/human_control.py
python -u scripts/train.py
