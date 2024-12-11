# Pyramid-Flow SD3
# pyramid_flow.py
import torch
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video
import gradio as gr

def generate_pyramid_video(prompt, num_frames):
    model_path = "path/to/downloaded/model"
    model = PyramidDiTForVideoGeneration(model_path, "bf16")
    model.enable_sequential_cpu_offload()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        frames = model.generate(prompt=prompt, num_inference_steps=[10] * 3, height=768, width=1280, temp=16)
    export_to_video(frames, "pyramid.mp4", fps=24)
    return "pyramid.mp4"

pyramid_interface = gr.Interface(
    fn=generate_pyramid_video,
    inputs=["text", gr.Slider(1, 100, step=1, label="Number of Frames")],
    outputs="video",
    title="Pyramid-Flow Video Generation"
)

if __name__ == "__main__":
    pyramid_interface.launch()
