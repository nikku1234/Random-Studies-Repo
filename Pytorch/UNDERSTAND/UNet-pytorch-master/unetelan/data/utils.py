import numpy as np


def read_ppm(path):
    """ Read file in ppm format

    Parameters
    ----------
    path : str
        path to the image
    """
    with open(path, 'rb') as f:
        img = f.readlines()

    img = b''.join(line for line in img if not line.strip().startswith(b'#'))
    header, dims, maxval, img = img.split(b'\n', maxsplit=3)

    ch = {b'P5': 1, b'P6': 3}[header]
    w, h = [int(item) for item in dims.decode().split()]
    maxval = int(maxval)

    arr = np.array(np.frombuffer(img, dtype=np.uint8))
    arr = arr.reshape(h, w, ch)

    arr = arr.squeeze()

    return arr
