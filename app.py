import gradio as gr
from core.utils import (
    example_file_path,
    _load_volume_from_any,
    volume_stats,
    browse_axis_fast,
    browse_overlay_axis_fast,
    segment_volume,
    APP_TMP_DIR,
    clean_temp,
    write_mask_tif,
)
import urllib.request
import time, threading, tempfile, os
from typing import Union
from gradio import skip


CLEAN_EVERY_SEC = 1800      # every 30 min
CLEAN_AGE_HOURS = 12        # every 12 hours

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

def get_axis_max(volume, axis):
    """Get the maximum index of each axis."""
    if volume is None:
        return 0
    shape = volume.shape
    return shape[{"Z": 0, "Y": 1, "X": 2}[axis]] - 1

def reset_app():
    """Reset everything to the initial state."""
    return (
        gr.update(value=None),   # file_input
        None,                    # volume_state
        None,                    # seg_state
        gr.update(visible=False),# group_input
        gr.update(visible=False),# segment_btn
        gr.update(value=0), gr.update(value=0), gr.update(value=0),
        gr.update(value=None), gr.update(value=None), gr.update(value=None),
        gr.update(visible=False),# group_seg
        gr.update(value=0), gr.update(value=0), gr.update(value=0),
        gr.update(value=None), gr.update(value=None), gr.update(value=None)
    )

def segment_api(file_obj: Union[dict, str, bytes]) -> gr.FileData:
    volume = _load_volume_from_any(file_obj)
    seg = segment_volume(volume)
    if seg is None:
        raise gr.Error("Segmentation failed")
    out_path = write_mask_tif(seg)

    return gr.FileData(
        path=out_path,
        orig_name=os.path.basename(out_path),
        mime_type="image/tiff",
    )

def run_seg_with_progress(volume, progress=gr.Progress(track_tqdm=True)):
    """Surface a progress bar in Gradio while the model runs."""
    if volume is None:
        return None
    progress(0.1, desc="Preparing model…")
    seg = segment_volume(volume)
    progress(1.0, desc="Done")
    return seg

with gr.Blocks(delete_cache=(1800, 21600)) as demo:
    # Expose ONLY the /segment API/MCP tool
    gr.api(
        segment_api,
        api_name="segment",
        api_description="Accepts a 3D TIF/TIFF (URL, uploaded file, or raw bytes) and returns a path to the compressed TIF mask."
    )

    # -------- UI --------
    gr.Markdown("# 🐭 3D Lungs Segmentation")
    gr.Markdown("### ⚠️ Note: the visualization may take some time to render!")

    # States
    last_url_state = gr.State("")   # last processed ?file_url
    volume_state = gr.State()
    seg_state = gr.State()
    norm_state = gr.State()

    file_input = gr.File(
        file_types=[".tif", ".tiff"],
        file_count="single",
        label="Upload your 3D TIF or TIFF file"
    )

    gr.Examples(
        examples=[[example_file_path]],
        inputs=[file_input],
        label="Try an example!",
        examples_per_page=1
    )

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

    # -------- Callbacks (hidden from API/MCP) --------
    file_input.change(
        fn=lambda f: _load_volume_from_any(f) if f is not None else skip(),
        inputs=file_input,
        outputs=volume_state,
        show_api=False
    ).then(
        fn=lambda vol: volume_stats(vol) if vol is not None else skip(),
        inputs=volume_state,
        outputs=norm_state,
        show_api=False
    ).then(
        fn=lambda vol: gr.update(visible=True) if vol is not None else skip(),
        inputs=volume_state,
        outputs=group_input,
        show_api=False
    ).then(
        fn=lambda vol: gr.update(visible=True) if vol is not None else skip(),
        inputs=volume_state,
        outputs=segment_btn,
        show_api=False
    ).then(
        fn=lambda vol: (
            gr.update(maximum=get_axis_max(vol, "Z")),
            gr.update(maximum=get_axis_max(vol, "Y")),
            gr.update(maximum=get_axis_max(vol, "X")),
        ) if vol is not None else (skip(), skip(), skip()),
        inputs=volume_state,
        outputs=[z_slider, y_slider, x_slider],
        show_api=False
    ).then(
        fn=lambda vol, st: (
            browse_axis_fast("Z", 0, vol, st),
            browse_axis_fast("Y", 0, vol, st),
            browse_axis_fast("X", 0, vol, st),
        ) if vol is not None else (skip(), skip(), skip()),
        inputs=[volume_state, norm_state],
        outputs=[z_img, y_img, x_img],
        show_api=False
    )

    z_slider.change(
        fn=lambda idx, vol, st: browse_axis_fast("Z", idx, vol, st),
        inputs=[z_slider, volume_state, norm_state],
        outputs=z_img,
        show_api=False
    )
    y_slider.change(
        fn=lambda idx, vol, st: browse_axis_fast("Y", idx, vol, st),
        inputs=[y_slider, volume_state, norm_state],
        outputs=y_img,
        show_api=False
    )
    x_slider.change(
        fn=lambda idx, vol, st: browse_axis_fast("X", idx, vol, st),
        inputs=[x_slider, volume_state, norm_state],
        outputs=x_img,
        show_api=False
    )

    segment_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(interactive=False)),
        inputs=[],
        outputs=[loading_md, segment_btn],
        show_api=False
    ).then(
        fn=run_seg_with_progress,
        inputs=volume_state,
        outputs=seg_state,
        show_api=False
    ).then(
        fn=lambda s: gr.update(visible=(s is not None)),
        inputs=seg_state,
        outputs=group_seg,
        show_api=False
    ).then(
        fn=lambda vol: (
            gr.update(maximum=get_axis_max(vol, "Z")),
            gr.update(maximum=get_axis_max(vol, "Y")),
            gr.update(maximum=get_axis_max(vol, "X")),
        ),
        inputs=volume_state,
        outputs=[z_slider_seg, y_slider_seg, x_slider_seg],
        show_api=False
    ).then(
        fn=lambda z, y, x, vol, seg, st: (
            browse_overlay_axis_fast("Z", z, vol, seg, st),
            browse_overlay_axis_fast("Y", y, vol, seg, st),
            browse_overlay_axis_fast("X", x, vol, seg, st),
        ),
        inputs=[z_slider_seg, y_slider_seg, x_slider_seg, volume_state, seg_state, norm_state],
        outputs=[z_img_overlay, y_img_overlay, x_img_overlay],
        show_api=False
    ).then(
        fn=lambda: (gr.update(visible=False), gr.update(interactive=True)),
        inputs=[],
        outputs=[loading_md, segment_btn],
        show_api=False
    )

    z_slider_seg.change(
        fn=lambda idx, vol, seg, st: browse_overlay_axis_fast("Z", idx, vol, seg, st),
        inputs=[z_slider_seg, volume_state, seg_state, norm_state],
        outputs=z_img_overlay,
        show_api=False
    )
    y_slider_seg.change(
        fn=lambda idx, vol, seg, st: browse_overlay_axis_fast("Y", idx, vol, seg, st),
        inputs=[y_slider_seg, volume_state, seg_state, norm_state],
        outputs=y_img_overlay,
        show_api=False
    )
    x_slider_seg.change(
        fn=lambda idx, vol, seg, st: browse_overlay_axis_fast("X", idx, vol, seg, st),
        inputs=[x_slider_seg, volume_state, seg_state, norm_state],
        outputs=x_img_overlay,
        show_api=False
    )

    reset_btn.click(
        fn=reset_app,
        inputs=[],
        outputs=[
            file_input,
            volume_state,
            seg_state,
            group_input,
            segment_btn,
            z_slider, y_slider, x_slider,
            z_img, y_img, x_img,
            group_seg,
            z_slider_seg, y_slider_seg, x_slider_seg,
            z_img_overlay, y_img_overlay, x_img_overlay
        ],
        show_api=False
    )


    # -------- URL loader --------
    @demo.load(
        inputs=[last_url_state],
        outputs=[last_url_state, file_input],  # only these two
        show_api=False
    )
    def load_from_query(prev_url, request: gr.Request):
        params = request.query_params
        url = params.get("file_url") or ""

        # No URL -> no-op
        if not url:
            return [gr.skip(), gr.skip()]

        # 🔧 Short-circuit: same URL as last time -> no-op
        if url == prev_url:
            return [gr.skip(), gr.skip()]

        # Download to CLOSED temp file and programmatically set the File value.
        fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=str(APP_TMP_DIR))
        os.close(fd)
        try:
            urllib.request.urlretrieve(url, tmp_path)
        except Exception as e:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise gr.Error(f"Failed to download file_url: {e}")

        return [url, gr.update(value=tmp_path)]


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=16).launch(mcp_server=True)