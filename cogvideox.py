# CogVideoX
# cogvideox.py
import time
import torch
from cogvideox.api.api import infer_forward_api
from cogvideox.ui.ui import ui_modelscope
import gradio as gr

def generate_cogvideox_video(prompt, num_frames):
    model_name = "path/to/model/CogVideoX-5b"
    model_type = "Control"
    demo, controller = ui_modelscope(model_name, model_type, "samples", False, torch.bfloat16)

    frames = infer_forward_api(None, prompt, num_frames=num_frames)
    export_to_video(frames, "cogvideox.mp4", fps=24)
    return "cogvideox.mp4"

cogvideox_interface = gr.Interface(
    fn=generate_cogvideox_video,
    inputs=["text", gr.Slider(1, 100, step=1, label="Number of Frames")],
    outputs="video",
    title="CogVideoX Video Generation"
)

if __name__ == "__main__":
    cogvideox_interface.launch()