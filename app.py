import gradio as gr
from core.utils import *

import urllib.request
import tempfile
import os, time, threading, atexit
from core.utils import APP_TMP_DIR, clean_temp, write_mask_tif

CLEAN_EVERY_SEC = 1800      # every 30 min
CLEAN_AGE_HOURS = 6         # every 6 hours

def _start_cleanup_daemon():
    def _loop():
        while True:
            try:
                clean_temp(CLEAN_AGE_HOURS)
            except Exception as e:
                print(f"[cleanup daemon] {e}")
            time.sleep(CLEAN_EVERY_SEC)
    threading.Thread(target=_loop, daemon=True).start()

_start_cleanup_daemon()
atexit.register(lambda: clean_temp(0))


def get_axis_max(volume, axis):
    """Get the maximum index of each axis."""
    if volume is None:
        return 0
    shape = volume.shape
    return shape[{"Z": 0, "Y": 1, "X": 2}[axis]] - 1

def reset_app():
    """Reset everything to the initial state."""
    return (
        gr.update(value=None),
        None,
        None,
        gr.update(visible=False),
        gr.update(value=0), gr.update(value=0), gr.update(value=0),
        gr.update(value=None), gr.update(value=None), gr.update(value=None),
        gr.update(visible=False),
        gr.update(value=0), gr.update(value=0), gr.update(value=0),
        gr.update(value=None), gr.update(value=None), gr.update(value=None)
    )

def segment_api(file_obj):
    """
    Accepts a TIF/TIFF via API, returns a TIF mask file path.
    """
    if not file_obj:
        raise gr.Error("No file provided")

    # Read volume (and let load_volume clean the temp upload)
    volume = load_volume(file_obj)
    seg = segment_volume(volume)  # uses your existing model wrapper
    if seg is None:
        raise gr.Error("Segmentation failed")

    # Write compressed TIF to app temp; return file path
    out_path = write_mask_tif(seg)
    return out_path

def run_seg_with_progress(volume, progress=gr.Progress(track_tqdm=True)):
    """
    Thin wrapper to surface a progress bar in Gradio while the model runs.
    """
    if volume is None:
        return None
    progress(0.1, desc="Preparing model…")
    seg = segment_volume(volume)  # existing function from utils.py
    progress(1.0, desc="Done")
    return seg

with gr.Blocks(delete_cache=(1800, 21600)) as demo:
    # ---- API (hidden) ----
    _api_in = gr.File(file_types=[".tif", ".tiff"], visible=False)
    _api_out = gr.File(visible=False)
    gr.Button(visible=False).click(
        fn=segment_api,
        inputs=_api_in,
        outputs=_api_out,
        api_name="segment"
    )

    # ---- UI ----
    gr.Markdown("# 🐭 3D Lungs Segmentation")
    gr.Markdown("### ⚠️ Note: the visualization may take some time to render!")

    volume_state = gr.State()
    seg_state = gr.State()
    norm_state = gr.State()

    file_input = gr.File(file_types=[".tif", ".tiff"], label="Upload your 3D TIF or TIFF file")

    # ---- Example loader ----
    gr.Examples(
        examples=[[example_file_path]],
        inputs=[file_input],
        label="Try an example!",
        examples_per_page=1
    )

    # ---- RAW SLICES VIEWER ----
    with gr.Group(visible=False) as group_input:
        gr.Markdown("### Raw Volume Slices")
        with gr.Row():
            z_slider = gr.Slider(0, 0, step=1, label="Z Slice")
            y_slider = gr.Slider(0, 0, step=1, label="Y Slice")
            x_slider = gr.Slider(0, 0, step=1, label="X Slice")
        with gr.Row():
            z_img = gr.Image(label="Z")
            y_img = gr.Image(label="Y")
            x_img = gr.Image(label="X")

    segment_btn = gr.Button("Segment", visible=False)

    loading_md = gr.Markdown("⏳ **Segmenting…** This can take a bit.", visible=False)

    # ---- OVERLAY SLICES VIEWER ----
    with gr.Group(visible=False) as group_seg:
        gr.Markdown("### Segmentation Overlay Slices")
        with gr.Row():
            z_slider_seg = gr.Slider(0, 0, step=1, label="Z Slice (Overlay)")
            y_slider_seg = gr.Slider(0, 0, step=1, label="Y Slice (Overlay)")
            x_slider_seg = gr.Slider(0, 0, step=1, label="X Slice (Overlay)")
        with gr.Row():
            z_img_overlay = gr.Image(label="Z + Mask")
            y_img_overlay = gr.Image(label="Y + Mask")
            x_img_overlay = gr.Image(label="X + Mask")

    reset_btn = gr.Button("Reset")

    gr.Markdown("#### 📝 This work is based on the Bachelor Project of Quentin Chappuis 2024; for more information, consult the [repository](https://github.com/qchapp/lungs-segmentation)!")

    # ---- CALLBACKS ----

    # A) Load volume
    file_input.change(
        fn=load_volume,
        inputs=file_input,
        outputs=volume_state
    ).then(
        fn=volume_stats,
        inputs=volume_state,
        outputs=norm_state
    ).then(
        fn=lambda vol: gr.update(visible=(vol is not None)),
        inputs=volume_state,
        outputs=group_input
    ).then(
        fn=lambda vol: gr.update(visible=(vol is not None)),
        inputs=volume_state,
        outputs=segment_btn
    ).then(
        fn=lambda vol: (
            gr.update(maximum=get_axis_max(vol, "Z")),
            gr.update(maximum=get_axis_max(vol, "Y")),
            gr.update(maximum=get_axis_max(vol, "X")),
        ),
        inputs=volume_state,
        outputs=[z_slider, y_slider, x_slider]
    ).then(
        fn=lambda vol, st: (
            browse_axis_fast("Z", 0, vol, st),
            browse_axis_fast("Y", 0, vol, st),
            browse_axis_fast("X", 0, vol, st),
        ),
        inputs=[volume_state, norm_state],
        outputs=[z_img, y_img, x_img]
    )

    # B) RAW sliders
    z_slider.change(fn=lambda idx, vol, st: browse_axis_fast("Z", idx, vol, st), inputs=[z_slider, volume_state, norm_state], outputs=z_img)
    y_slider.change(fn=lambda idx, vol, st: browse_axis_fast("Y", idx, vol, st), inputs=[y_slider, volume_state, norm_state], outputs=y_img)
    x_slider.change(fn=lambda idx, vol, st: browse_axis_fast("X", idx, vol, st), inputs=[x_slider, volume_state, norm_state], outputs=x_img)

    # C) Segment
    segment_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(interactive=False)),
        inputs=[],
        outputs=[loading_md, segment_btn]
    ).then(
        fn=run_seg_with_progress,  # <— shows a progress bar
        inputs=volume_state,
        outputs=seg_state
    ).then(
        fn=lambda s: gr.update(visible=(s is not None)),
        inputs=seg_state,
        outputs=group_seg
    ).then(
        fn=lambda vol: (
            gr.update(maximum=get_axis_max(vol, "Z")),
            gr.update(maximum=get_axis_max(vol, "Y")),
            gr.update(maximum=get_axis_max(vol, "X")),
        ),
        inputs=volume_state,
        outputs=[z_slider_seg, y_slider_seg, x_slider_seg]
    ).then(
        fn=lambda z, y, x, vol, seg, st: (
            browse_overlay_axis_fast("Z", z, vol, seg, st),
            browse_overlay_axis_fast("Y", y, vol, seg, st),
            browse_overlay_axis_fast("X", x, vol, seg, st),
        ),
        inputs=[z_slider_seg, y_slider_seg, x_slider_seg, volume_state, seg_state, norm_state],
        outputs=[z_img_overlay, y_img_overlay, x_img_overlay]
    ).then(
        fn=lambda: (gr.update(visible=False), gr.update(interactive=True)),
        inputs=[],
        outputs=[loading_md, segment_btn]
    )

    # D) OVERLAY sliders
    z_slider_seg.change(fn=lambda idx, vol, seg, st: browse_overlay_axis_fast("Z", idx, vol, seg, st), inputs=[z_slider_seg, volume_state, seg_state, norm_state], outputs=z_img_overlay)
    y_slider_seg.change(fn=lambda idx, vol, seg, st: browse_overlay_axis_fast("Y", idx, vol, seg, st), inputs=[y_slider_seg, volume_state, seg_state, norm_state], outputs=y_img_overlay)
    x_slider_seg.change(fn=lambda idx, vol, seg, st: browse_overlay_axis_fast("X", idx, vol, seg, st), inputs=[x_slider_seg, volume_state, seg_state, norm_state], outputs=x_img_overlay)

    # E) Reset
    reset_btn.click(
        fn=reset_app,
        inputs=[],
        outputs=[
            file_input,
            volume_state,
            seg_state,
            group_input,
            z_slider, y_slider, x_slider,
            z_img, y_img, x_img,
            group_seg,
            z_slider_seg, y_slider_seg, x_slider_seg,
            z_img_overlay, y_img_overlay, x_img_overlay
        ]
    )

    # ---- HANDLE QUERY PARAMETERS ----
    @demo.load(
        outputs=[
            file_input,
            volume_state,
            norm_state,
            group_input,
            segment_btn,
            z_slider, y_slider, x_slider,
            z_img, y_img, x_img
        ]
    )
    def load_from_query(request: gr.Request):
        params = request.query_params

        if "file_url" in params:
            try:
                # A) Download the file from the URL to a managed temporary path
                url = params["file_url"]
                fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=str(APP_TMP_DIR))
                os.close(fd)
                urllib.request.urlretrieve(url, tmp_path)

                # B) Open the file as a binary object
                with open(tmp_path, "rb") as f:
                    volume = load_volume(f)

                # Remove downloaded temp file now that it's in memory
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    print(f"[load_from_query] couldn't remove {tmp_path}: {e}")

                # C) Return values for all components
                stats = volume_stats(volume)
                return [
                    gr.update(value=None),
                    volume,
                    stats,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(maximum=get_axis_max(volume, "Z")),
                    gr.update(maximum=get_axis_max(volume, "Y")),
                    gr.update(maximum=get_axis_max(volume, "X")),
                    browse_axis_fast("Z", 0, volume, stats),
                    browse_axis_fast("Y", 0, volume, stats),
                    browse_axis_fast("X", 0, volume, stats),
                ]

            except Exception as e:
                print(f"[Error loading file_url] {e}")

        # Fallback if no file_url or failure
        return [
            None,
            None,
            (0.0, 1.0),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(maximum=0),
            gr.update(maximum=0),
            gr.update(maximum=0),
            None, None, None
        ]


if __name__ == "__main__":
    try:
        demo.queue(concurrency_count=1, max_size=16).launch()
    except TypeError:
        try:
            demo.queue(max_size=16).launch()
        except TypeError:
            demo.queue().launch()