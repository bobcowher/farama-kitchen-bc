"""The one place camera frames get resized.

The recorder archives at the render size, the loader reduces shards to whatever
the training arena declares, and a rollout has to build its image the same way.
If those disagree the policy just gets quietly worse -- nothing raises, nothing
logs -- so they all come through here.

Only integer reductions are allowed, and that restriction is the point. PIL
BOX, cv2 INTER_AREA and torch interpolate(mode='area') all claim to average
over the source footprint, but they only agree when the ratio divides evenly.
Measured on the kitchen scene against PIL BOX, worst-pixel difference:

    640 -> 320  (2x)     cv2  1    torch  1
    640 -> 160  (4x)     cv2  1    torch  1
    640 -> 128  (5x)     cv2  1    torch  1
    640 -> 256  (2.5x)   cv2 47    torch 77
    640 -> 224  (2.857x) cv2 42    torch 53

At an integer ratio every output pixel is exactly an NxN block and there is
nothing left to disagree about; the 1 is uint8 rounding. At a fractional ratio
BOX weights each source pixel by its fractional coverage while torch's adaptive
pooling averages an integer window unweighted. Keeping ratios integral means a
resize done anywhere -- on the GPU in a training loop, by a model's own image
processor -- still matches this one, so correctness stops depending on everyone
remembering to call the same function.

That is why the collector renders at 896: 896 = 2^7 * 7, so it reduces cleanly
to 224 and 448 (PaliGemma, OpenVLA and pi0 all take 224) as well as 128 and 112.
640 reaches none of those.

Rendering natively at the smaller size is also not equivalent to rendering large
and reducing here. Native 256 against 640-then-resize differs by 1.5/255 on
average, but the flattest half of the image is identical at 0.09 while the
edgiest tenth carries 48% of the error at 7.3. MuJoCo rasterizes one sample per
pixel, so at 256 an edge either covers the pixel center or does not; reducing
from 640 averages ~6.25 rendered samples and reconstructs partial coverage.
Edges are most of what a conv net's first layer keys on, so eval renders large
and comes through here.
"""

import numpy as np
from PIL import Image

# The collector renders at RENDER_SIZE, and these are the sizes a shard can be
# reduced to: its even divisors, dropping the ones too small to be images (896
# also divides by 7, 14, 28 and 56). Every entry is an integer reduction, which
# is the property that keeps resize backends agreeing -- see above. 224 is the
# VLM target, 128 the small-CNN one, 896 a pass-through.
#
# These two belong to each other. Changing the render size means recomputing
# this list, not just editing the number.
RENDER_SIZE = 896
SUPPORTED_SIZES = (64, 112, 128, 224, 448, 896)


def resize(frames, size):
    """Reduce camera frames to size x size. Accepts (H,W,3) or (N,H,W,3).

    BOX averages over the source pixels each output pixel covers. Upsampling
    and fractional ratios both raise -- see the module docstring for why the
    ratio has to be integral.
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

    # Two different mistakes, so two different messages. Reporting the second
    # as "must be one of [... 448 ...]" while rejecting 448 reads as a
    # contradiction and sends you looking at the wrong number.
    if size not in SUPPORTED_SIZES:
        raise ValueError(
            f"{size} is not a supported size. Must be one of "
            f"{list(SUPPORTED_SIZES)}.")

    if width % size:
        raise ValueError(
            f"{size} is a supported size but does not divide {width}px frames "
            f"evenly. SUPPORTED_SIZES describes the {RENDER_SIZE}px this project "
            f"renders at, so {width}px frames came from somewhere else -- an "
            f"image_size that was not updated, or shards from an older render.")

    # PIL crawls on the flipped view MuJoCo hands back (negative row stride),
    # 2.8 ms against 0.8 for the same result. The copy costs 0.02 ms.
    frames = np.ascontiguousarray(frames)

    out = np.empty((len(frames), size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(frame).resize((size, size), Image.BOX))
    return out[0] if single else out
