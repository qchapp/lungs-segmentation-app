import numpy as np
import tifffile
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

def browse_all_axes(relative_idx, volume):
    """Return raw Z/Y/X slices at relative index."""
    if volume is None:
        return None, None, None

    z, y, x = volume.shape
    idx_z = int(relative_idx * (z - 1))
    idx_y = int(relative_idx * (y - 1))
    idx_x = int(relative_idx * (x - 1))

    slice_z = _to_8bit(volume[idx_z])
    slice_y = _to_8bit(volume[:, idx_y, :])
    slice_x = _to_8bit(volume[:, :, idx_x])

    return (
        Image.fromarray(slice_z),
        Image.fromarray(slice_y),
        Image.fromarray(slice_x)
    )

def browse_overlay_all_axes(relative_idx, volume, seg):
    """Return overlay Z/Y/X slices at relative index."""
    if volume is None or seg is None:
        return None, None, None

    z, y, x = volume.shape
    idx_z = int(relative_idx * (z - 1))
    idx_y = int(relative_idx * (y - 1))
    idx_x = int(relative_idx * (x - 1))

    slices = [
        (volume[idx_z], seg[idx_z]),
        (volume[:, idx_y, :], seg[:, idx_y, :]),
        (volume[:, :, idx_x], seg[:, :, idx_x]),
    ]

    result = []
    for raw_slice, seg_slice in slices:
        raw_8bit = _to_8bit(raw_slice)
        raw_rgb = np.stack([raw_8bit] * 3, axis=-1)
        mask_rgb = np.zeros_like(raw_rgb)
        mask_rgb[..., 0] = (seg_slice * 255).astype(np.uint8)

        alpha = 0.3
        blended = (1 - alpha) * raw_rgb + alpha * mask_rgb
        result.append(Image.fromarray(blended.astype(np.uint8)))

    return tuple(result)
