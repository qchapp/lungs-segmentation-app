import gradio as gr
from core.utils import *

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

with gr.Blocks() as demo:
    gr.Markdown("# 🐭 3D Lungs Segmentation")
    gr.Markdown("### ⚠️ Note: the visualization may take some time to render!")

    volume_state = gr.State()
    seg_state = gr.State()

    file_input = gr.File(file_types=[".tif", ".tiff"], label="Upload your 3D TIF or TIFF file")

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

    gr.Markdown("#### 📝 This work is based on the Bachelor Thesis of Quentin Chappuis 2024; for more information, consult the [repository](https://github.com/qchapp/lungs-segmentation)!")

    # ---- CALLBACKS ----

    # A) Load volume
    file_input.change(
        fn=load_volume,
        inputs=file_input,
        outputs=volume_state
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
        fn=lambda vol: (
            browse_axis("Z", 0, vol),
            browse_axis("Y", 0, vol),
            browse_axis("X", 0, vol),
        ),
        inputs=volume_state,
        outputs=[z_img, y_img, x_img]
    )

    # B) RAW sliders
    z_slider.change(fn=lambda idx, vol: browse_axis("Z", idx, vol), inputs=[z_slider, volume_state], outputs=z_img)
    y_slider.change(fn=lambda idx, vol: browse_axis("Y", idx, vol), inputs=[y_slider, volume_state], outputs=y_img)
    x_slider.change(fn=lambda idx, vol: browse_axis("X", idx, vol), inputs=[x_slider, volume_state], outputs=x_img)

    # C) Segment
    segment_btn.click(
        fn=segment_volume,
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
        fn=lambda z, y, x, vol, seg: (
            browse_overlay_axis("Z", z, vol, seg),
            browse_overlay_axis("Y", y, vol, seg),
            browse_overlay_axis("X", x, vol, seg),
        ),
        inputs=[z_slider_seg, y_slider_seg, x_slider_seg, volume_state, seg_state],
        outputs=[z_img_overlay, y_img_overlay, x_img_overlay]
    )

    # D) OVERLAY sliders
    z_slider_seg.change(fn=lambda idx, vol, seg: browse_overlay_axis("Z", idx, vol, seg), inputs=[z_slider_seg, volume_state, seg_state], outputs=z_img_overlay)
    y_slider_seg.change(fn=lambda idx, vol, seg: browse_overlay_axis("Y", idx, vol, seg), inputs=[y_slider_seg, volume_state, seg_state], outputs=y_img_overlay)
    x_slider_seg.change(fn=lambda idx, vol, seg: browse_overlay_axis("X", idx, vol, seg), inputs=[x_slider_seg, volume_state, seg_state], outputs=x_img_overlay)

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

if __name__ == "__main__":
    demo.launch()