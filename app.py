# Main App Loader
# app.py
import subprocess
import gradio as gr

def load_model_and_generate_video(model_name, prompt):
    if model_name == "Mochi 1 Preview":
        subprocess.Popen(["python", "mochi1_preview.py"])
    elif model_name == "Pyramid-Flow SD3":
        subprocess.Popen(["python", "pyramid_flow.py"])
    elif model_name == "CogVideoX":
        subprocess.Popen(["python", "cogvideox.py"])
    return f"Generating video using {model_name} with prompt: '{prompt}'..."

app = gr.Interface(
    fn=load_model_and_generate_video,
    inputs=[
        gr.Dropdown(["Mochi 1 Preview", "Pyramid-Flow SD3", "CogVideoX"], label="Select Model"),
        gr.Textbox(label="Enter Prompt for Video Generation")
    ],
    outputs="text",
    title="Video Generation Loader"
)

if __name__ == "__main__":
    app.launch()
