import gradio as gr

from core.utils import *

def update_slider_range_for_volume(volume):
    """Update the first slider's range based on the volume shape."""
    if volume is None:
        return gr.update(minimum=0, maximum=0, value=0, visible=False)
    return gr.update(minimum=0, maximum=volume.shape[0] - 1, value=0, visible=True)

def update_slider_range_for_overlay(seg, volume):
    """
    Update the second slider's range based on the segmentation shape.
    We'll use the same Z dimension as the volume or seg.
    """
    if seg is None or volume is None:
        return gr.update(minimum=0, maximum=0, value=0, visible=False)
    return gr.update(minimum=0, maximum=volume.shape[0] - 1, value=0, visible=True)

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
        gr.update(minimum=0, maximum=0, value=0, visible=False),
        gr.update(value=None),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(minimum=0, maximum=0, value=0, visible=False),
        gr.update(value=None)
    )

# Main app
with gr.Blocks() as demo:
    # Title of the page
    gr.Markdown("# 3D Lungs Segmentation")

    volume_state = gr.State()
    seg_state = gr.State()

    # (1) File input
    file_input = gr.File(
        file_types=[".tif", ".tiff"],
        label="Upload your 3D TIF or TIFF file!"
    )

    # (2) File uploaded viewer
    with gr.Group(visible=False) as group_input:
        gr.Markdown("### View the Raw Volume")
        z_slider_input = gr.Slider(
            minimum=0, 
            maximum=0,
            step=1,
            value=0,
            label="Raw Volume Z-Slice",
            visible=False
        )
        raw_slice_img = gr.Image(label="Raw Slice")

    # (3) Segment button (hidden until file is uploaded)
    segment_btn = gr.Button("Segment", visible=False)

    # (4) Segmentation overlay viewer
    with gr.Group(visible=False) as group_seg:
        gr.Markdown("### View the Overlay (Raw + Mask)")
        z_slider_seg = gr.Slider(
            minimum=0,
            maximum=0,
            step=1,
            value=0,
            label="Overlay Volume Z-Slice",
            visible=False
        )
        overlay_slice_img = gr.Image(label="Overlay Slice")

    # (5) Reset button
    reset_btn = gr.Button("Reset", visible=True)

    # ---- CALL-BACKS ----
    #
    # A) On file upload -> load volume -> store in volume_state
    #    then update slider range -> show group input -> show segmentation button
    file_input.change(
        fn=load_volume,
        inputs=file_input,
        outputs=volume_state
    ).then(
        fn=update_slider_range_for_volume,
        inputs=volume_state,
        outputs=z_slider_input
    ).then(
        fn=lambda vol: gr.update(visible=(vol is not None)),
        inputs=volume_state,
        outputs=group_input
    ).then(
        fn=lambda vol: gr.update(visible=(vol is not None)),
        inputs=volume_state,
        outputs=segment_btn
    )

    # B) On z_slider_input change -> show raw slice
    z_slider_input.change(
        fn=browse_raw_slice,
        inputs=[z_slider_input, volume_state],
        outputs=raw_slice_img
    )

    # C) On "Segment" -> segment_volume -> store in seg_state
    #    then update second slider range -> show group_seg
    segment_btn.click(
        fn=segment_volume,
        inputs=volume_state,
        outputs=seg_state
    ).then(
        fn=update_slider_range_for_overlay,
        inputs=[seg_state, volume_state],
        outputs=z_slider_seg
    ).then(
        fn=lambda s: gr.update(visible=(s is not None)),
        inputs=seg_state,
        outputs=group_seg
    )

    # D) On z_slider_seg change -> show overlay slice
    z_slider_seg.change(
        fn=browse_overlay_slice,
        inputs=[z_slider_seg, volume_state, seg_state],
        outputs=overlay_slice_img
    )

    # E) Reset everything when clicked
    reset_btn.click(
        fn=reset_app,
        inputs=[],
        outputs=[
            file_input,
            volume_state,
            seg_state,
            group_input,
            z_slider_input,
            raw_slice_img,
            segment_btn,
            group_seg,
            z_slider_seg,
            overlay_slice_img
        ]
    )

if __name__ == "__main__":
    demo.launch()