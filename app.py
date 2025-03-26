import gradio as gr
from core.utils import *

def reset_app():
    """
    Reset everything to the initial state:
      - Clear file input
      - Clear volume_state, seg_state
      - Hide both groups
      - Hide the segment button
      - Reset sliders + images
    """
    return (
        gr.update(value=None),
        None,
        None,
        gr.update(visible=False),
        gr.update(value=0.5),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(visible=False),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
    )

with gr.Blocks() as demo:
    gr.Markdown("# 3D Lungs Segmentation")

    volume_state = gr.State()
    seg_state = gr.State()

    file_input = gr.File(file_types=[".tif", ".tiff"], label="Upload your 3D TIF or TIFF file")

    # --- Raw Slice Viewer ---
    with gr.Group(visible=False) as group_input:
        gr.Markdown("### Raw Slices (Z / Y / X)")
        rel_slider = gr.Slider(0, 1, step=0.01, value=0.5, label="Relative Slice Index")
        with gr.Row():
            img_z = gr.Image(label="Z")
            img_y = gr.Image(label="Y")
            img_x = gr.Image(label="X")

    segment_btn = gr.Button("Segment", visible=False)

    # --- Overlay Viewer ---
    with gr.Group(visible=False) as group_seg:
        gr.Markdown("### Segmentation Overlay (Z / Y / X)")
        with gr.Row():
            img_z_overlay = gr.Image(label="Z + Mask")
            img_y_overlay = gr.Image(label="Y + Mask")
            img_x_overlay = gr.Image(label="X + Mask")

    reset_btn = gr.Button("Reset")

    # A) On file upload → load → update state → show viewer → trigger image view
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
        fn=browse_all_axes,
        inputs=[rel_slider, volume_state],
        outputs=[img_z, img_y, img_x]
    )

    # B) Slider changes raw slices
    rel_slider.change(
        fn=browse_all_axes,
        inputs=[rel_slider, volume_state],
        outputs=[img_z, img_y, img_x]
    )

    # C) Segment → store state → show overlays
    segment_btn.click(
        fn=segment_volume,
        inputs=volume_state,
        outputs=seg_state
    ).then(
        fn=lambda s: gr.update(visible=(s is not None)),
        inputs=seg_state,
        outputs=group_seg
    ).then(
        fn=browse_overlay_all_axes,
        inputs=[rel_slider, volume_state, seg_state],
        outputs=[img_z_overlay, img_y_overlay, img_x_overlay]
    )

    # D) Slider changes overlays too
    rel_slider.change(
        fn=browse_overlay_all_axes,
        inputs=[rel_slider, volume_state, seg_state],
        outputs=[img_z_overlay, img_y_overlay, img_x_overlay]
    )

    # E) Reset everything
    reset_btn.click(
        fn=reset_app,
        inputs=[],
        outputs=[
            file_input,
            volume_state,
            seg_state,
            group_input,
            rel_slider,
            img_z,
            img_y,
            img_x,
            group_seg,
            img_z_overlay,
            img_y_overlay,
            img_x_overlay
        ]
    )

if __name__ == "__main__":
    demo.launch()