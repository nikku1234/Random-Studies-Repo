import moviepy.editor as mpy
import numpy as np
from PIL import Image
from torch.autograd import Variable as Var
from visdom import Visdom
import tempfile
import torch as th


def scalar(s):
    if isinstance(s, Var):
        s = s.data

    if type(s) not in {int, float}:
        s = s[0]

    return s


def arr(s):
    return np.array([s])


def to_numpy(arr, numpy=True):
    if isinstance(arr, Var):
        arr = arr.data

    if arr.__class__.__module__ == 'torch':
        arr = arr.cpu().type(th.FloatTensor)

        if numpy:
            arr = arr.numpy()

    return arr


def update_trace(vis, win, name, x, y, title=""):
    x, y = arr(scalar(x)), arr(scalar(y))

    if win is None:
        win = vis.line(
            X=x,
            Y=y,
            opts={
                'legend': [name],
                'title': title
            }
        )
    else:
        vis.updateTrace(
            X=x,
            Y=y,
            win=win,
            name=name,
        )

    return win


def video(vis: Visdom, images, masks, win=None, fps=4, title=""):
    I = to_numpy(images)
    M = to_numpy(masks)
    V = np.concatenate([I, M], axis=2) * 255

    arrs = [V[i, ...] for i in range(V.shape[0])]
    imgs = [Image.fromarray(arr).convert("RGB").resize((512, 256)) for arr in arrs]

    imgs = [np.asarray(img) for img in imgs]

    ID    = next(tempfile._get_candidate_names())
    fname = f"/tmp/clip_{ID}.mp4"

    clip = mpy.ImageSequenceClip(imgs, fps)
    clip.write_videofile(fname, fps=fps, audio=False, verbose=False, progress_bar=False)

    vis.video(videofile=fname, win=win,
              opts={
                  'title': title,
                  'fps': fps,
                  'height': 270,
                  'width': 530
              })
