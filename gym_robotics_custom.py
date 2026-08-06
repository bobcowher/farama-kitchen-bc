import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, Wrapper


class HeldSetpointWrapper(Wrapper):
    """Stops the arm from sagging under gravity.

    FrankaKitchen reads the action as a joint velocity and integrates it into a
    position setpoint for the actuators -- but it integrates onto the *measured*
    qpos every step (see _ctrl_velocity_limits in franka_env.py). Gravity droops
    each joint slightly below its setpoint, that droop gets read back in, and the
    next setpoint starts from the drooped position, so the arm ratchets downward:
    ~61 degrees of drift over 200 steps of zero action.

    Holding the setpoint we actually commanded turns that drift into a bounded
    ~1 degree offset, which is what a position servo should do.

    max_lead caps how far the setpoint may run ahead of the arm. Without it the
    setpoint outruns the joints during a fast move and they keep coasting for
    ~20 steps after you release the stick. It must stay above the servo's own
    steady-state droop (~0.02 rad) or the clamp re-anchors to measured qpos and
    the sag comes back. Measured at 0.05: sag 1.4 deg, coast 2.9 deg.
    """

    def __init__(self, env, max_lead=0.05):
        super().__init__(env)
        self.max_lead = max_lead

    def step(self, action):
        result = self.env.step(action)
        env = self.unwrapped
        # data.ctrl is the setpoint the env just commanded, post-clipping.
        qpos = env.data.qpos[:9]
        setpoint = np.clip(env.data.ctrl[:9], qpos - self.max_lead, qpos + self.max_lead)
        env.robot_env._last_robot_qpos = setpoint.copy()
        return result


class RoboGymObservationWrapper(ObservationWrapper):

    def __init__(self, env, goal='microwave'):
        super(RoboGymObservationWrapper, self).__init__(env)
        self.goal = goal

    def set_goal(self, goal):
        self.goal = goal

    def reset(self):
        observation, info = self.env.reset()
        observation = self.process_observation(observation)
        return observation, info
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        observation = self.process_observation(observation)
        return observation, reward, done, truncated, info

    def process_observation(self, observation):
        obs_pos = observation['observation']
        obs_achieved_goal = observation['achieved_goal']
        obs_desired_goal = observation['desired_goal']

        # print(obs_achieved_goal)

        obs_concatenated = np.concatenate((obs_pos, obs_achieved_goal[self.goal], obs_desired_goal[self.goal]))

        return obs_concatenated

