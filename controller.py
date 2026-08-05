import numpy as np
import pygame

# FrankaKitchen-v1 action dims, from the MuJoCo actuator list:
#   0-6  panda0_joint1 .. panda0_joint7
#   7    r_gripper_finger_joint
#   8    l_gripper_finger_joint
# Zero holds position, so releasing everything is a no-op rather than a snap home.

# Indices below were measured on an 8BitDo pad in Switch mode, reporting as
# "Pro Controller" (14 buttons, 4 axes, 1 hat). Changing the pad's input mode
# renumbers the buttons, so these would need re-measuring.
#
# Sticks report up as negative and right as positive on both sticks.
#
# (axis index, action dim, sign)
AXIS_MAP = (
    (0, 0, -1.0),  # left stick horizontal  -> joint1
    (1, 1, -1.0),  # left stick vertical    -> joint2  (up = +)
    (3, 2, +1.0),  # right stick vertical   -> joint3  (up = -, opposite of left)
    (2, 3, -1.0),  # right stick horizontal -> joint4
)

# button -> (action dim, value). Exactly one owner per dim per direction.
BUTTON_MAP = {
    0: (4, -1.0),  # B  -> joint5 -
    2: (4, +1.0),  # X  -> joint5 +
    6: (5, -1.0),  # R  -> joint6 -  wrist side-to-side tilt
    5: (5, +1.0),  # L  -> joint6 +  wrist side-to-side tilt
    7: (6, -1.0),  # L2 -> joint7 -
    8: (6, +1.0),  # R2 -> joint7 +
}

# button -> gripper_closed state.
GRIPPER_MAP = {
    1: True,   # A -> close
    3: False,  # Y -> open
}

# Unbound and free: button 4 (unidentified), 9 (Minus), 10 (Plus), 11-13,
# and the whole D-pad, which reports as hat 0 and is not read at all.

AXIS_DEADZONE = 0.1


class Controller:
    def __init__(self):
        self.gripper_closed = None

        pygame.init()
        pygame.joystick.init()

        # Assuming only one joystick is connected
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def get_action(self):
        """
        Map PlayStation controller input to the robot's action space.

        Returns None when nothing is being pressed, which tells the caller to
        skip the env step rather than advance the sim with a zero action.
        """
        pygame.event.pump()

        action = np.zeros(9)

        for axis, dim, sign in AXIS_MAP:
            value = self.joystick.get_axis(axis) * sign
            if abs(value) >= AXIS_DEADZONE:
                action[dim] = value

        pressed = [b for b in range(self.joystick.get_numbuttons())
                   if self.joystick.get_button(b)]

        # Accumulate so opposing directions cancel instead of one winning.
        for button in pressed:
            if button in BUTTON_MAP:
                dim, value = BUTTON_MAP[button]
                action[dim] += value
                print(f"Button {button} pressed -> action[{dim}] {value:+.1f}")

        gripper_button_pressed = False
        for button in pressed:
            if button in GRIPPER_MAP:
                self.gripper_closed = GRIPPER_MAP[button]
                gripper_button_pressed = True
                print(f"Button {button} pressed -> gripper "
                      f"{'close' if self.gripper_closed else 'open'}")

        if not action.any() and not gripper_button_pressed:
            return None

        # Applied last and from stored state, so the gripper holds its position
        # across frames and combines with arm motion.
        if self.gripper_closed is True:
            action[7] = -1.0  # Close gripper
            action[8] = -1.0
        elif self.gripper_closed is False:
            action[7] = 1.0  # Open gripper
            action[8] = 1.0

        return np.clip(action, -1.0, 1.0)
