# Mochi 1 Preview
# mochi1_preview.py
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import gradio as gr

def generate_mochi_video(prompt, num_frames):
    pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    with torch.autocast("cuda", torch.bfloat16):
        frames = pipe(prompt, num_frames=num_frames).frames[0]
    export_to_video(frames, "mochi.mp4", fps=30)
    return "mochi.mp4"

mochi_interface = gr.Interface(
    fn=generate_mochi_video,
    inputs=["text", gr.Slider(1, 100, step=1, label="Number of Frames")],
    outputs="video",
    title="Mochi Video Generation"
)

if __name__ == "__main__":
    mochi_interface.launch()