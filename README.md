# Cheers-HF-Demo

A Hugging Face Gradio Demo for the **ai9stars/Cheers** Vision-Language Model.

## Overview
This repository contains a full-stack Gradio web application designed to showcase the capabilities of the `ai9stars/Cheers` model. The app provides an interactive interface for image-to-text generation tasks, allowing users to upload images and prompt the model for descriptions or visual question answering.

## Features
- Interactive Web Interface: Powered by Gradio for a seamless user experience.
- Image-to-Text Capabilities: Describe images or answer questions based on visual input.
- High Performance: Optimized model loading with `bfloat16` precision for modern GPUs.
- Seamless Hugging Face Integration: Direct usage of `transformers` for AutoModelForCausalLM and AutoProcessor.

## Prerequisites
Ensure you have the following installed:
- Python 3.10 or higher
- pip (version >= 26.0.0, as per `pre-requirements.txt`)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Cheers-HF-Demo.git
cd Cheers-HF-Demo
```

2. Upgrade pip:
```bash
pip install -r pre-requirements.txt
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
The main dependencies include:
- `torch` (2.8.0)
- `torchvision`
- `transformers` (4.51.3)
- `gradio`
- `Pillow`
- `einops`
- `scipy`
- `matplotlib`
- `black`
- `param`

## Usage

To start the Gradio application locally, run:

```bash
python app.py
```

This will launch a local server. Open the provided URL (usually `http://127.0.0.1:7860`) in your web browser.

## Model Details
- Checkpoint: `ai9stars/Cheers`
- Architecture: Causal Language Model configured for vision-text inputs.
- Evaluation Mode: The model is loaded in evaluation mode by default and offloads to CUDA if a compatible GPU is present.

## License
Refer to the `LICENSE.txt` file for terms and conditions.

---
Note: Make sure your environment has a CUDA-compatible GPU for optimal performance, though CPU fallback is supported.
