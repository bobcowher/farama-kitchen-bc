"""Watch a collected demo: the camera frames in a window, the joint states in the CLI.

    python view_run.py dataset/hinge_cabinet/shard_2026_08_06_21_42_43.npz

space  play/pause      left/right (or a/d)  step one frame
r      restart         q or Esc             quit
"""

import os
import sys

import cv2
import numpy as np

FPS = 20  # matches the collector's time.sleep(0.05)

# cv2 reports arrow keys differently depending on the GUI backend, so accept
# every encoding we might see, plus a/d as a guaranteed fallback.
PREV_KEYS = {65361, 2424832, 81, ord('a')}
NEXT_KEYS = {65363, 2555904, 83, ord('d')}
QUIT_KEYS = {ord('q'), 27}


def row(label, values):
    return f"  {label:<10s}" + " ".join(f"{v:7.3f}" for v in values)


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {os.path.basename(__file__)} <shard.npz>")

    path = sys.argv[1]
    if not os.path.exists(path):
        sys.exit(f"no such file: {path}")

    if not os.environ.get("DISPLAY"):
        sys.exit("DISPLAY is not set, so there is no window to draw into.")

    data = np.load(path)
    frames = data["camera_scene"]
    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    action = data["action"]
    reward = data["reward"]
    done = data["done"]
    n = len(frames)

    # A shard is one episode, so at most one step can be terminal. More than
    # that means this file predates the fix and holds the old SAC bootstrap
    # mask, which is the inverse of a done flag.
    legacy_done = done.sum() > 1
    done_label = "done(raw)" if legacy_done else "done"

    print("=" * 72)
    print(f"file        {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"task        {data['task_name']}")
    print(f"instruction {data['task_description']}")
    print(f"steps       {n}   frame {frames.shape[1]}x{frames.shape[2]}")
    print(f"reward      total {reward.sum():.1f}, "
          f"earned at step(s) {list(np.flatnonzero(reward > 0))}")
    if legacy_done:
        print()
        print(f"NOTE: {done.sum()} of {n} steps are flagged terminal, so this shard")
        print("predates the done fix and stores the old SAC bootstrap mask")
        print("(1 = keep going, 0 = terminal). Read the raw value inverted.")
    print("=" * 72)
    print("space play/pause   left/right (or a/d) step   r restart   q quit")
    print()

    window = os.path.basename(path)
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    i = 0
    playing = True
    shown = -1

    while True:
        if i != shown:
            frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            tag = f"{i + 1}/{n}   reward {reward[i]:.2f}"
            if reward[i] > 0:
                tag += "   SUCCESS"
            # Drawn twice so the text stays readable over a light kitchen.
            cv2.putText(frame, tag, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, tag, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, frame)

            print(f"step {i + 1}/{n}   reward {reward[i]:.3f}   "
                  f"{done_label} {bool(done[i])}")
            print(row("joint_pos", joint_pos[i]))
            print(row("joint_vel", joint_vel[i]))
            print(row("action", action[i]), flush=True)
            shown = i

        at_end = playing and i == n - 1
        if at_end:
            playing = False
            print("-- end of run, press r to restart --")

        key = cv2.waitKeyEx(int(1000 / FPS) if playing else 0)

        if key in QUIT_KEYS:
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('r'):
            i, playing = 0, True
        elif key in PREV_KEYS:
            i, playing = max(0, i - 1), False
        elif key in NEXT_KEYS:
            i, playing = min(n - 1, i + 1), False
        elif playing:
            i = min(n - 1, i + 1)

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
