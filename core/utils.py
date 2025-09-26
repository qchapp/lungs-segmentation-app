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

APP_TMP_DIR = Path(tempfile.gettempdir()) / "lungs_seg_tmp"
APP_TMP_DIR.mkdir(parents=True, exist_ok=True)

def new_tmp_path(basename: str = "tmp.tif") -> str:
    """Return a unique path inside the app temp dir."""
    uid = uuid.uuid4().hex[:8]
    return str(APP_TMP_DIR / f"{uid}_{basename}")

def clean_temp(max_age_hours: float = 6.0) -> None:
    cutoff = time.time() - max_age_hours * 3600 if max_age_hours > 0 else float("inf")
    protected = PROTECTED_PATHS
    for p in APP_TMP_DIR.glob("*"):
        try:
            rp = p.resolve()
            if rp in protected:
                continue
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

def load_volume(file_obj):
    if not file_obj:
        return None
    path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or file_obj
    arr = tifffile.imread(path)

    try:
        if path and os.path.exists(path):
            src = Path(path).resolve()
            if src not in PROTECTED_PATHS:
                os.remove(src)
    except Exception as e:
        print(f"[load_volume] couldn't remove temp file {path}: {e}")

    return arr

def segment_volume(volume):
    """Run segmentation on the loaded volume (return shape (Z, Y, X))."""
    if volume is None:
        return None
    return model.segment_lungs(volume)

# Optimization for faster processing
def volume_stats(volume):
    """Return (min, max) as floats for global 8-bit scaling."""
    if volume is None:
        return (0.0, 1.0)
    return float(volume.min()), float(volume.max())

def _to_8bit_stats(arr, mn, mx):
    rng = max(mx - mn, 1e-8)
    return np.clip((arr - mn) / rng * 255.0, 0, 255).astype(np.uint8)

def browse_axis_fast(axis, idx, volume, stats):
    """Same as browse_axis but uses precomputed global stats."""
    if volume is None:
        return None
    mn, mx = stats
    if axis == "Z":
        slice_ = volume[idx]
    elif axis == "Y":
        slice_ = volume[:, idx, :]
    elif axis == "X":
        slice_ = volume[:, :, idx]
    else:
        return None
    return Image.fromarray(_to_8bit_stats(slice_, mn, mx))

def browse_overlay_axis_fast(axis, idx, volume, seg, stats, alpha=0.35):
    """Overlay using global stats (fewer allocations, faster)."""
    if volume is None or seg is None:
        return None
    mn, mx = stats
    if axis == "Z":
        raw = volume[idx];        mask = seg[idx]
    elif axis == "Y":
        raw = volume[:, idx, :];  mask = seg[:, idx, :]
    elif axis == "X":
        raw = volume[:, :, idx];  mask = seg[:, :, idx]
    else:
        return None

    raw8 = _to_8bit_stats(raw, mn, mx)
    rgb  = np.repeat(raw8[..., None], 3, axis=-1)
    # color mask in red channel
    mask_rgb = np.zeros_like(rgb)
    mask_rgb[..., 0] = (mask.astype(np.uint8) * 255)

    blended = rgb.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha
    return Image.fromarray(blended.astype(np.uint8))

# Example file
def get_example_file():
    url = "https://zenodo.org/record/8099852/files/lungs_ct.tif?download=1"
    tmp_path = APP_TMP_DIR / "example_lungs.tif"
    if not tmp_path.exists():
        urllib.request.urlretrieve(url, tmp_path)
    return str(tmp_path)

example_file_path = get_example_file()
PROTECTED_PATHS = {Path(example_file_path).resolve()}