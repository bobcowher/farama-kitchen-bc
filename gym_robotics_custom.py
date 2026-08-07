import numpy as np
from gymnasium import ObservationWrapper, Wrapper, spaces
from PIL import Image


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


class VLAObservationWrapper(ObservationWrapper):
    """A camera view plus proprioception, for a vision-language-action policy.

    Returns {"camera_scene", "joint_pos", "joint_vel"}, matching the
    camera/joint_pos/joint_vel split used by the other buffers in this repo.

    Object poses are deliberately left out. The old wrapper concatenated the
    env's achieved_goal and desired_goal onto a flat joint-state vector, which
    made sense when the policy had no eyes. A VLA reads the scene from the image
    and the task from the language instruction, so both were redundant anyway:
    achieved_goal is a copy of columns already inside `observation`, and
    desired_goal is a per-task constant. They were also the only reason the
    observation width varied by task (61/63/73), which blocked mixing shards.

    camera_scene is the free camera from DEFAULT_CAMERA_CONFIG -- the same view
    the human window shows, so demonstrator and policy see the same thing. The
    model's own 'left_cap' and 'right_cap' cameras point at a countertop and a
    cabinet face; neither shows the arm, so neither is used.

    There is no wrist view. One was built and aimed (see the wrist-camera
    branch), but this environment is a fixed scene with a fixed viewpoint and
    coarse tasks -- doors, knobs, a kettle -- so the conditions that make
    eye-in-hand views pay off in the literature mostly do not apply here. It
    doubled storage for a benefit this setup was unlikely to collect.
    """

    def __init__(self, env, image_size=640):
        super().__init__(env)
        self.image_size = image_size
        self._set_render_size(env.unwrapped, image_size)
        self.observation_space = spaces.Dict({
            "camera_scene": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
            "joint_pos": spaces.Box(-np.inf, np.inf, (9,), np.float32),
            "joint_vel": spaces.Box(-np.inf, np.inf, (9,), np.float32),
        })

    def _set_render_size(self, kitchen, image_size):
        """Raise the render resolution above the env's 480 default.

        Must run before anything renders, because the viewers read these when
        they are first created.
        """
        robot = kitchen.robot_env
        renderer = robot.mujoco_renderer
        assert not renderer._viewers, (
            "render size must be set before the first render")

        # The offscreen viewer sizes itself from renderer.width/height, not from
        # offwidth, so both have to be raised or we would upscale a 480 render.
        # This also enlarges the teleop window, which is no bad thing.
        robot.width = robot.height = image_size
        renderer.width = renderer.height = image_size
        kitchen.model.vis.global_.offwidth = image_size
        kitchen.model.vis.global_.offheight = image_size

        self._renderer = renderer

    def _grab(self):
        # Renders offscreen through a second viewer, so this works even while
        # render_mode='human' is driving the teleop window.
        self._renderer.camera_id = -1  # free camera, per DEFAULT_CAMERA_CONFIG
        frame = self._renderer.render("rgb_array")
        if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
            frame = np.asarray(Image.fromarray(frame).resize(
                (self.image_size, self.image_size), Image.BILINEAR))
        return frame

    def observation(self, observation):
        robot_obs = observation["observation"]
        return {
            "camera_scene": self._grab(),
            "joint_pos": robot_obs[:9].astype(np.float32),
            "joint_vel": robot_obs[9:18].astype(np.float32),
        }

