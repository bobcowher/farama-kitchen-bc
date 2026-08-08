"""The one place camera frames get resized.

The recorder writes 640x640 shards, the loader downsamples them to whatever
the training arena declares, and a rollout has to hand the policy an image
built the same way. If those three ever disagree the policy just gets quietly
worse -- nothing raises, nothing logs -- so they all call this.

Rendering natively at the smaller size is *not* equivalent to rendering large
and resizing here. Measured on the kitchen scene, native 256 against
640-then-resize: mean absolute difference 1.5/255, but the flattest half of
the image is identical (0.09) while the edgiest tenth carries half the total
error (7.3). MuJoCo rasterizes one sample per pixel, so at 256 an edge either
covers the pixel center or does not; resizing from 640 averages ~6.25 rendered
samples per output pixel, which reconstructs partial coverage. Edges are most
of what a conv net's first layer keys on, so eval renders large and comes
through here rather than rendering small.
"""

import numpy as np
from PIL import Image


def resize(frames, size):
    """Resize camera frames to size x size. Accepts (H,W,3) or (N,H,W,3).

    BOX averages over the source pixels each output pixel covers, which is what
    you want going 640 -> 256: bilinear samples too sparsely at that ratio and
    aliases high-frequency detail into the only input the policy gets.

    Upsampling raises rather than happening. It would invent detail, and every
    caller reaching this point with an undersized frame has a real problem: a
    Dataset arena wider than its shards, or a render that ignored the size it
    was given.
    """
    single = frames.ndim == 3
    if single:
        frames = frames[None]

    width = frames.shape[1]
    if width == size:
        return frames[0] if single else frames

    if width < size:
        raise ValueError(
            f"frames are {width}px but {size}px was asked for. That would mean "
            f"upsampling; work at {width}px or smaller.")

    out = np.empty((len(frames), size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(frame).resize((size, size), Image.BOX))
    return out[0] if single else out
