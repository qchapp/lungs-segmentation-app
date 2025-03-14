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

def browse_raw_slice(z_idx, volume):
    """
    Show just the raw slice at z_idx.
    """
    if volume is None:
        return None
    z_dim = volume.shape[0]
    z_idx = max(0, min(z_idx, z_dim - 1))

    # Slice the volume
    raw_slice = volume[z_idx]
    # Convert raw slice to 8-bit grayscale
    raw_8bit = _to_8bit(raw_slice)
    return Image.fromarray(raw_8bit)

def browse_overlay_slice(z_idx, volume, seg):
    """
    Show an overlay of the raw slice + mask in red for the given z_idx.
    """
    if volume is None or seg is None:
        return None
    z_dim = volume.shape[0]
    z_idx = max(0, min(z_idx, z_dim - 1))

    # Slice the volume
    raw_slice = volume[z_idx]
    seg_slice = seg[z_idx]  # 0 or 1

    # Convert raw slice to 8-bit grayscale
    raw_8bit = _to_8bit(raw_slice)
    # Make 3-channel RGB
    raw_rgb = np.stack([raw_8bit, raw_8bit, raw_8bit], axis=-1)

    # Create a red mask for seg=1
    mask_rgb = np.zeros_like(raw_rgb)
    mask_rgb[..., 0] = (seg_slice * 255).astype(np.uint8)

    # Alpha-blend
    alpha = 0.3
    blended = (1 - alpha) * raw_rgb + alpha * mask_rgb
    blended = blended.astype(np.uint8)

    return Image.fromarray(blended)