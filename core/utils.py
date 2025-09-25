import numpy as np
import tifffile
import os
import tempfile
import urllib.request
from PIL import Image
from pathlib import Path
import time, uuid, atexit
from unet_lungs_segmentation import LungsPredict

model = LungsPredict()

APP_TMP_DIR = Path(os.environ.get("APP_TMP_DIR", Path(tempfile.gettempdir()) / "lungs_seg_tmp"))
APP_TMP_DIR.mkdir(parents=True, exist_ok=True)

def new_tmp_path(basename: str = "tmp.tif") -> str:
    """Return a unique path inside the app temp dir."""
    uid = uuid.uuid4().hex[:8]
    return str(APP_TMP_DIR / f"{uid}_{basename}")

def clean_temp(max_age_hours: float = 6.0) -> None:
    """Delete old files in our app temp dir (keeps the example by name)."""
    cutoff = time.time() - max_age_hours * 3600 if max_age_hours > 0 else float("inf")
    for p in APP_TMP_DIR.glob("*"):
        try:
            if p.name == "example_lungs.tif":
                continue  # don't delete the cached example
            if max_age_hours == 0 or p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
        except Exception as e:
            print(f"[cleanup] could not remove {p}: {e}")

atexit.register(lambda: clean_temp(0))  # purge on shutdown

def write_mask_tif(mask: np.ndarray) -> str:
    """Write a mask volume to a compressed TIFF in app temp and return the path."""
    out_path = new_tmp_path("mask.tif")
    tifffile.imwrite(out_path, mask.astype(np.uint8), compression="zlib")
    return out_path

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
    """Read the uploaded TIF as a NumPy array (Z, Y, X) and clean the temp upload."""
    if not file_obj:
        return None

    # Gradio File can be a file-like with .name or a path-like
    path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or file_obj
    arr = tifffile.imread(path)

    # Remove source temp file to avoid disk growth (but keep the cached example)
    try:
        if path and os.path.exists(path) and os.path.basename(path) != "example_lungs.tif":
            os.remove(path)
    except Exception as e:
        print(f"[load_volume] couldn't remove temp file {path}: {e}")

    return arr


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
    tmp_path = APP_TMP_DIR / "example_lungs.tif"
    if not tmp_path.exists():
        urllib.request.urlretrieve(url, tmp_path)
    return str(tmp_path)

example_file_path = get_example_file()