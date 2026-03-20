# Cheers-HF-Demo

The Cheers-HF-Demo is a Gradio-based web interface engineered to serve the ai9stars/Cheers Vision-Language Model. This application acts as a direct demonstration of high-performance multimodal inference capabilities, abstracting the underlying complexity of PyTorch tensor operations and Transformers library implementations into an accessible user interface. By integrating the AutoModelForCausalLM and AutoProcessor architectures, the system executes sophisticated image-to-text generation and visual question answering (VQA) pipelines. The model instantiation process strictly enforces torch.bfloat16 precision and leverages dynamic hardware offloading to available CUDA-compatible devices, ensuring optimized memory allocation and minimized latency during evaluation passes. This repository provides a scalable baseline for deploying advanced multimodal models in real-world diagnostic, analytical, and generative contexts.

## Architecture and Execution Pipeline
The core execution logic resides within `app.py`, which handles the initialization sequence for the `ai9stars/Cheers` checkpoint. The application utilizes a customized causal language modeling configuration specifically tuned for processing interleaved visual and textual input tensors. The pipeline automatically manages dynamic input scaling, tokenization of natural language prompts, and the generation of text sequences utilizing optimized decoding strategies inherent to the Transformers framework.

## Technical Specifications
- Multimodal Processing Engine: Implements native support for synchronous processing of high-resolution image tensors and complex string sequences, translating cross-modal inputs into deterministic text outputs.
- Memory and Compute Optimization: The enforcement of `bfloat16` precision reduces the VRAM requirement by approximately 50% compared to standard float32 precision, mitigating out-of-memory (OOM) exceptions without a statistically significant degradation in generation quality.
- Hardware Abstraction Layer: The application includes an automated device detection mechanism that binds model execution to `cuda` if an NVIDIA GPU is available, with an automatic fallback mechanism to `cpu` execution.
- Frontend Abstraction: The Gradio interface maps directly to Python backend functions, converting base64 encoded image uploads and text field inputs into PyTorch-compatible formats before executing the forward pass of the model.

## Prerequisites and Environment Setup
Deployment of this application requires adherence to the following system baseline:
- Operating System: Linux, Windows, or macOS.
- Runtime Environment: Python >= 3.10.
- Package Management: `pip` version >= 26.0.0 (strictly enforced via `pre-requirements.txt` to mitigate dependency resolution failures).

## Installation Protocol

To initialize a local instance, execute the following commands to clone the repository and resolve dependencies:

1. Clone the repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Cheers-HF-Demo.git
cd Cheers-HF-Demo
```

2. Resolve pre-installation dependencies:
```bash
pip install -r pre-requirements.txt
```

3. Install core dependencies:
```bash
pip install -r requirements.txt
```

### Dependency Stack
The execution environment relies on a specific version matrix. Critical dependencies include `torch` (2.8.0) and `torchvision` for accelerated tensor computations, `transformers` (4.51.3) for model definition and tokenizer handling, and `gradio` for the web server and UI bindings. Supporting libraries such as `Pillow`, `einops`, `scipy`, and `matplotlib` are utilized for image preprocessing and matrix reshaping operations required by the vision encoder.

## Deployment Execution

Initiate the application server by executing the primary script:

```bash
python app.py
```

The script will bind a local server process, outputting an accessible loopback address (e.g., `http://127.0.0.1:7860`).

## Model Configuration Details
- Checkpoint Retrieval: The system resolves and downloads the `ai9stars/Cheers` weights directly from the Hugging Face Model Hub.
- Remote Code Execution: Due to non-standard architectural modifications in the vision-language projection layers, instantiation requires the `trust_remote_code=True` parameter.
- Graph Evaluation Mode: The application enforces `model.eval()` post-load to disable stochastic operations such as dropout, ensuring reproducible inference graphs.

## License and Compliance
Usage of this codebase is governed by the terms outlined in the `LICENSE.txt` file. 
