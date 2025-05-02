import numpy as np
import tifffile
import os
import tempfile
import urllib.request
from PIL import Image
from unet_lungs_segmentation import LungsPredict

model = LungsPredict()

def _to_8bit(arr):
    """Convert float/int array to 8-bit [0..255]."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    rng = mx - mn
    if rng < 1e-8:
        rng = 1.0
    norm = (arr - mn) / rng
    return (norm * 255).astype(np.uint8)

def load_volume(file_obj):
    """Read the uploaded TIF as a NumPy array (Z, Y, X)."""
    if not file_obj:
        return None
    return tifffile.imread(file_obj.name)

def segment_volume(volume):
    """Run segmentation on the loaded volume (return shape (Z, Y, X))."""
    if volume is None:
        return None
    return model.segment_lungs(volume)

def browse_axis(axis, idx, volume):
    """Return a single raw slice for the given axis."""
    if volume is None:
        return None

    if axis == "Z":
        slice_ = volume[idx]
    elif axis == "Y":
        slice_ = volume[:, idx, :]
    elif axis == "X":
        slice_ = volume[:, :, idx]
    else:
        return None

    return Image.fromarray(_to_8bit(slice_))

def browse_overlay_axis(axis, idx, volume, seg):
    """Return a single overlay slice for the given axis."""
    if volume is None or seg is None:
        return None

    if axis == "Z":
        raw = volume[idx]
        mask = seg[idx]
    elif axis == "Y":
        raw = volume[:, idx, :]
        mask = seg[:, idx, :]
    elif axis == "X":
        raw = volume[:, :, idx]
        mask = seg[:, :, idx]
    else:
        return None

    raw_8bit = _to_8bit(raw)
    raw_rgb = np.stack([raw_8bit] * 3, axis=-1)
    mask_rgb = np.zeros_like(raw_rgb)
    mask_rgb[..., 0] = (mask * 255).astype(np.uint8)

    alpha = 0.3
    blended = (1 - alpha) * raw_rgb + alpha * mask_rgb
    return Image.fromarray(blended.astype(np.uint8))

# Example file
def get_example_file():
    url = "https://zenodo.org/record/8099852/files/lungs_ct.tif?download=1"
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "example_lungs.tif")

    # Only download if it doesn't already exist
    if not os.path.exists(tmp_path):
        urllib.request.urlretrieve(url, tmp_path)

    return tmp_path

example_file_path = get_example_file()