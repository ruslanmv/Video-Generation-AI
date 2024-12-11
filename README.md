# Video Generation AI

## Overview
This repository contains a Gradio-based application for generating videos using three state-of-the-art text-to-video models:

1. **Mochi 1 Preview** (genmo/mochi-1-preview)
2. **Pyramid-Flow SD3** (rain1011/pyramid-flow-sd3)
3. **CogVideoX** (THUDM/CogVideoX-5b)

The application allows users to generate videos with prompts by selecting the desired model and specifying parameters such as the number of frames.

---

## Project Structure

- `mochi1_preview.py`: Script for generating videos using the Mochi 1 Preview model.
- `pyramid_flow.py`: Script for generating videos using the Pyramid-Flow SD3 model.
- `cogvideox.py`: Script for generating videos using the CogVideoX model.
- `app.py`: Main application script to select and launch any of the above models through a Gradio interface.

---

## Prerequisites

### Python Environment
- Python 3.8 or higher
- Recommended: Create a virtual environment for dependencies.

### Hardware Requirements
- A GPU with at least 20GB VRAM is recommended for high-quality video generation.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ruslanmv/video-generation-ai.git
   cd video-generation-ai
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install FFMPEG for video processing (if not already installed):
   ```bash
   sudo apt install ffmpeg   # Linux
   brew install ffmpeg       # macOS
   choco install ffmpeg      # Windows
   ```

---

## Setting Up Models

### Mochi 1 Preview

1. Download the weights:
   ```bash
   git clone https://github.com/genmoai/models
   cd models
   python scripts/download_weights.py <path_to_downloaded_directory>
   ```

2. Verify installation with Diffusers:
   ```bash
   pip install git+https://github.com/huggingface/diffusers.git
   ```

### Pyramid-Flow SD3

1. Create a new conda environment:
   ```bash
   conda create -n pyramid python=3.8.10
   conda activate pyramid
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download("rain1011/pyramid-flow-sd3", local_dir="<model_path>")
   ```

### CogVideoX

1. Install dependencies:
   ```bash
   pip install cogvideox
   ```

2. Set up the UI and inference:
   ```bash
   python cogvideox.py
   ```

---

## Running the Application

To launch the main app:

```bash
python app.py
```

Select one of the models (`Mochi 1 Preview`, `Pyramid-Flow SD3`, or `CogVideoX`) from the dropdown menu to start generating videos.

---

## Usage

1. Provide a text prompt describing the video you want to generate.
2. Adjust parameters like the number of frames.
3. Click "Submit" to generate and download the video.

---

## Example Prompts

- "A movie trailer featuring the adventures of a space explorer."
- "A close-up of a chameleon changing colors."

---

## Acknowledgments

Special thanks to:

- [Genmo Team](https://github.com/genmoai) for Mochi 1 Preview.
- [Rain1011](https://huggingface.co/rain1011) for Pyramid-Flow SD3.
- [THUDM Team](https://huggingface.co/THUDM) for CogVideoX.

---

## License
This project is licensed under the Apache 2.0 License.

