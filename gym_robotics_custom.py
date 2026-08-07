import mujoco
import numpy as np
from gymnasium import ObservationWrapper, Wrapper, spaces
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
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


# Wrist camera geometry, all in the panda0_link7 frame. The gripper points along
# +Z: hand at z=0.107, fingers at z=0.1654, TCP at z=0.210.
#
# Measured mesh extents of the arm's own bodies in that frame:
#     link7 disc   z 0.052..0.107   max radius 0.090
#     hand         z 0.081..0.173   max radius 0.104
#     fingers      z 0.166..0.260   max radius 0.032
#
# That 0.104 radius is why earlier guesses at this pose all failed. Anything
# mounted closer to the axis than the hand envelope is looking into the back of
# the hand, so the fingers never appear no matter how the camera is rotated.
# The camera has to sit outside the hand, not behind it.
#
# The fingers slide along the hand frame's Y axis, which chain.xml rotates -45
# degrees about link7's Z. Mounting on the perpendicular axis puts the camera
# looking between the fingers rather than through one of them.
_HAND_YAW = -np.pi / 4  # hand/finger frame rotation about link7 Z, per chain.xml
WRIST_MOUNT_AXIS = np.array([np.cos(_HAND_YAW), np.sin(_HAND_YAW), 0.0])
WRIST_CAM_RADIUS = 0.12  # just clear of the 0.104 hand envelope
WRIST_CAM_HEIGHT = 0.06
WRIST_CAM_TARGET = np.array([0.0, 0.0, 0.28])  # just past the fingertips at 0.260
WRIST_CAM_FOVY = 75


def _look_at(eye, target, up):
    """Camera quaternion aiming `eye` at `target`.

    MuJoCo cameras look down their own -Z with +Y as image up, so `up` must stay
    well away from the view axis or the frame is ill-conditioned -- picking it
    parallel to the aim direction is an easy way to get a silently wrong pose.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    z = -forward
    x = np.cross(up, z)
    norm = np.linalg.norm(x)
    assert norm > 0.2, f"up vector too close to the view axis (|x| = {norm:.3f})"
    x = x / norm
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    return quat


class VLAObservationWrapper(ObservationWrapper):
    """Two camera views plus proprioception, for a vision-language-action policy.

    Returns {"camera_scene", "camera_wrist", "joint_pos", "joint_vel"}, matching
    the camera/joint_pos/joint_vel split used by the other buffers in this repo.

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

    camera_wrist does not exist in the shipped model, so it is injected here.
    """

    def __init__(self, env, image_size=640):
        super().__init__(env)
        self.image_size = image_size
        self._add_wrist_camera(env.unwrapped, image_size)
        self.observation_space = spaces.Dict({
            "camera_scene": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
            "camera_wrist": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
            "joint_pos": spaces.Box(-np.inf, np.inf, (9,), np.float32),
            "joint_vel": spaces.Box(-np.inf, np.inf, (9,), np.float32),
        })

    def _add_wrist_camera(self, kitchen, image_size):
        """Recompile the model with a camera on the wrist and swap it in.

        Cameras cannot be added to an already-compiled MjModel, and the asset
        XML lives in site-packages, so the model is rebuilt from spec here.
        Adding a camera changes no state layout -- nq, nv and nu are untouched --
        so swapping model and data on the live env is safe. It has to happen
        before anything renders, because the viewers capture model/data when
        they are first created.
        """
        robot = kitchen.robot_env
        renderer = robot.mujoco_renderer
        assert not renderer._viewers, (
            "wrist camera must be injected before the first render")

        spec = mujoco.MjSpec.from_file(robot.fullpath)
        cam_pos = (WRIST_MOUNT_AXIS * WRIST_CAM_RADIUS
                   + np.array([0.0, 0.0, WRIST_CAM_HEIGHT]))
        cam = spec.body("panda0_link7").add_camera()
        cam.name = "wrist"
        cam.pos = cam_pos
        cam.quat = _look_at(cam_pos, WRIST_CAM_TARGET, WRIST_MOUNT_AXIS)
        cam.fovy = WRIST_CAM_FOVY

        model = spec.compile()
        # The offscreen viewer sizes itself from renderer.width/height, not from
        # offwidth, so both have to be raised or we would upscale a 480 render.
        # This also enlarges the teleop window, which is no bad thing.
        robot.width = robot.height = image_size
        renderer.width = renderer.height = image_size
        model.vis.global_.offwidth = image_size
        model.vis.global_.offheight = image_size
        data = mujoco.MjData(model)

        robot.model, robot.data = model, data
        kitchen.model, kitchen.data = model, data
        renderer.model, renderer.data = model, data
        robot.model_names = MujocoModelNames(model)

        self._renderer = renderer
        self._scene_cam = -1  # free camera, per DEFAULT_CAMERA_CONFIG
        self._wrist_cam = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")

    def _grab(self, camera_id):
        # Renders offscreen through a second viewer, so this works even while
        # render_mode='human' is driving the teleop window.
        self._renderer.camera_id = camera_id
        frame = self._renderer.render("rgb_array")
        if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
            frame = np.asarray(Image.fromarray(frame).resize(
                (self.image_size, self.image_size), Image.BILINEAR))
        return frame

    def observation(self, observation):
        robot_obs = observation["observation"]
        return {
            "camera_scene": self._grab(self._scene_cam),
            "camera_wrist": self._grab(self._wrist_cam),
            "joint_pos": robot_obs[:9].astype(np.float32),
            "joint_vel": robot_obs[9:18].astype(np.float32),
        }

