# Install required packages
#!pip install accelerate torch gradio transformers git+https://github.com/huggingface/diffusers sentencepiece opencv-python

import os

# Define a fallback for environments without GPU
if os.environ.get("SPACES_ZERO_GPU") is not None:
    import spaces
else:
    class spaces:
        @staticmethod
        def GPU(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import gradio as gr

# Application configuration
TITLE = "AI Video Generator 🌟"
DESCRIPTION = """\
🌈 Transform your imagination into stunning videos using advanced AI technology with Mochi-1-preview.\
Experience the magic of generative art! 🎥
"""
BUY_ME_A_COFFEE = """
<a href="https://buymeacoffee.com/ruslanmv" target="_blank">
    <button style="background-color: #FFDD00; border: none; color: black; 
                    padding: 10px 20px; text-align: center; 
                    text-decoration: none; display: inline-block; 
                    font-size: 16px; margin: 4px 2px; cursor: pointer; 
                    border-radius: 10px;">\
        ☕ Buy Me a Coffee
    </button>
</a>
"""
MODEL_PRE_TRAINED_ID = "genmo/mochi-1-preview"
EXAMPLES = [
    [
        "A colossal griffin perched atop a crumbling gothic castle, its golden wings outstretched against a blood-red sunset.  Below, a raging battle between knights and goblins unfolds amidst the ruins of a once-great city.  The air is filled with the clash of steel, the cries of the wounded, and the roar of the griffin's echoing cry.",
        90,
        30,
    ],
    [
        "A serene mountaintop monastery above the clouds, with monks practicing \"\
        Tai Chi at sunrise. The scene is filled with golden sunlight and \"\
        swirling mist, as cherry blossoms fall gently in the breeze.",
        70,
        24,
    ],
    [
        "An enchanted meadow where unicorns graze among glowing wildflowers. \"\
        Wisps of light float in the air, and a sparkling waterfall cascades into \"\
        a crystal-clear pond surrounded by colorful butterflies.",
        60,
        25,
    ],
    [
        "A sprawling underwater utopia with bioluminescent architecture, giant \"\
        jellyfish drifting gracefully, and schools of exotic fish weaving \"\
        through coral tunnels. The city is alive with vibrant marine life.",
        80,
        30,
    ],
    [
        "A vast alien desert with shimmering sands of gold and silver, \"\
        punctuated by colossal crystal spires. Twin suns set in the distance, \"\
        casting long, surreal shadows across the dunes.",
        75,
        28,
    ],
]

# Load the pre-trained model
pipe = DiffusionPipeline.from_pretrained(
    MODEL_PRE_TRAINED_ID, variant="bf16", torch_dtype=torch.bfloat16
)

# Enable memory-saving optimizations
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

@spaces.GPU(duration=60 * 3)
def generate_video(prompt, num_frames=84, fps=30, high_quality=False):
    """Generate a video based on the input prompt."""
    if high_quality:
        print("High quality option selected. Requires 42GB VRAM.")
        if os.environ.get("SPACES_ZERO_GPU") is not None:
            raise RuntimeError("High quality option may fail on ZeroGPU environments.")
        with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
            frames = pipe(prompt, num_frames=num_frames).frames[0]
    else:
        print("Standard quality option selected.")
        frames = pipe(prompt, num_frames=num_frames).frames[0]

    video_path = "generated_video.mp4"
    export_to_video(frames, video_path, fps=fps)
    return video_path

# Define the Gradio interface
interface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter a vivid text prompt... 🔍"),
        gr.Slider(minimum=1, maximum=240, value=84, label="Frames 🎥"),
        gr.Slider(minimum=1, maximum=60, value=30, label="FPS (Frames Per Second) ⏱"),
        gr.Checkbox(label="High Quality (Requires 42GB VRAM) 🛠"),
    ],
    outputs=gr.Video(label="Generated Video"),
    title=TITLE,
    description=DESCRIPTION,
    examples=EXAMPLES,
    article=BUY_ME_A_COFFEE,
)

# Apply custom CSS for better alignment
interface.css = """
.interface-title {
    text-align: center;
    font-size: 2em;
    color: #4A90E2;
    font-family: 'Arial', sans-serif;
}
.interface-description {
    text-align: center;
    font-size: 1.2em;
    color: #333333;
    margin-bottom: 20px;
}
"""

# Launch the Gradio application
if __name__ == "__main__":
    interface.launch(ssr_mode=False)
