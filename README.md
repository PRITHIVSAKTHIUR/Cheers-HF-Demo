# Cheers-HF-Demo

The Cheers-HF-Demo is an advanced, highly optimized full-stack web application built on the Gradio framework, engineered to interface seamlessly with the ai9stars/Cheers multimodal checkpoint. This repository provides a robust operational frontend that exposes the extensive capabilities of a state-of-the-art Vision-Language Model (VLM). Designed for high-performance execution environments, the application natively supports complex generative tasks spanning Text-to-Image synthesis, Image-to-Text comprehension, and direct Text-to-Text language modeling. By abstracting the intricacies of tensor manipulation, custom prompt templating, and hardware offloading, this repository serves as a foundational blueprint for deploying large-scale multimodal architectures into interactive, user-facing environments.

## Core Architecture and Execution Pipeline
The computational backbone of this application relies on the Hugging Face `transformers` library, explicitly utilizing the `AutoModelForCausalLM` and `AutoProcessor` classes. The `ai9stars/Cheers` checkpoint inherently dictates a causal language modeling approach, augmented with specialized projection layers to handle interleaved visual and textual modalities.

- Multimodal Modality Switching: The inference pipeline is dynamically routed based on user-selected execution modes.
  - Text-to-Image Synthesis: Translates natural language prompts into latent representations, which are subsequently decoded into pixel space utilizing classifier-free guidance (`cfg_scale`), stochastic sampling (`temperature`), and explicit inference step definitions (`num_inference_steps`).
  - Image-to-Text (VQA / Captioning): Ingests base64-encoded image payloads, decodes them into PIL objects, and constructs tensor representations via the `AutoProcessor`. The model contextualizes the visual data against natural language queries to generate deterministic textual responses.
  - Text-to-Text Modeling: Operates as a standard autoregressive language model when visual inputs are absent, leveraging optimized ChatML-style prompt templating logic.

- Precision and Memory Constraints: Model instantiation is strictly clamped to `torch.bfloat16` precision. This configuration significantly reduces the active GPU VRAM footprint required to hold the model weights and runtime activations, preventing out-of-memory (OOM) exceptions without introducing precision-loss artifacts during the forward pass.
- Hardware Abstraction and Acceleration: The runtime environment automatically detects and binds to CUDA-enabled devices (`device = "cuda"`). Furthermore, the core inference loop is decorated with `@spaces.GPU`, guaranteeing seamless compatibility and dynamic GPU allocation when deployed within Hugging Face Spaces (ZeroGPU architecture).

## Technical Specifications and Hyperparameter Controls
The application exposes a granular set of generation hyperparameters to the end user, ensuring absolute control over the inference behavior:
- `temperature`: Modulates the probability distribution over the vocabulary logic; lower values enforce greedy decoding, while higher values introduce stochasticity.
- `max_length`: Defines the absolute token limit for the autoregressive generation sequence.
- `cfg_scale`: (Specific to visual synthesis) Adjusts the classifier-free guidance scale, dictating how strictly the generated image adheres to the conditioning prompt.
- `num_inference_steps`: Determines the iterative denoising steps during visual generation, balancing computational cost against output fidelity.
- `seed` and `randomize_seed`: Exposes manual control over the underlying PyTorch pseudorandom number generator (PRNG) state to ensure deterministic output generation for rigorous evaluation and reproducibility.

## Prerequisites and Deployment Environment
To successfully deploy the Cheers-HF-Demo, the host environment must comply with the following technical baseline:
- Operating System compatibility across Linux, Windows, or macOS.
- Python Runtime Environment version >= 3.10.
- A functional Python Package Installer (`pip`), strictly upgraded to version >= 26.0.0. This mandate is enforced via `pre-requirements.txt` to circumvent dependency resolution graph failures prevalent in highly inter-dependent scientific stacks.

## Comprehensive Installation Protocol

Execute the following terminal commands to retrieve the source code, prepare the environment, and resolve all necessary library dependencies:

1. Retrieve the Source Repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Cheers-HF-Demo.git
cd Cheers-HF-Demo
```

2. Resolve Core Package Manager Dependencies:
```bash
pip install -r pre-requirements.txt
```

3. Install the Execution Stack:
```bash
pip install -r requirements.txt
```

### Dependency Matrix Analysis
The execution layer relies on tightly coupled versions of foundational deep learning libraries:
- `torch` (2.8.0): Provides the core tensor operations and automatic differentiation engine.
- `torchvision`: Handles specific image transformations and tensor standardizations.
- `transformers` (4.51.3): Manages the downloading, caching, and execution of the model graph and tokenizer.
- `gradio`: Constructs the asynchronous web socket layer and responsive frontend interface.
- Additional utilities such as `Pillow` (image decoding), `einops` (tensor reshaping), and `scipy` (scientific computing arrays) are required for internal pipeline operations.

## Execution and Local Deployment

To initialize the internal ASGI server and bind the application to a local port, execute the orchestrator script:

```bash
python app.py
```

Upon successful compilation of the frontend assets and loading of the model checkpoint into VRAM, the terminal will expose a local loopback address (e.g., `http://127.0.0.1:7860`).

## Security and Operational Caveats
- Remote Code Execution Risk: The instantiation of the `ai9stars/Cheers` checkpoint necessitates the flag `trust_remote_code=True`. This is due to the presence of custom Python logic within the model repository that defines the non-standard multimodal projection architecture. Ensure you are deploying within a secure or sandboxed environment.
- Evaluation Graph State: The script explicitly calls `model.eval()` post-initialization. This locks the computation graph, disabling dynamic operations such as Dropout and Batch Normalization, which is critical for consistent inference metrics.

## Licensing
The distribution, modification, and deployment of this codebase are strictly governed by the constraints defined in the `LICENSE.txt` document provided in the root directory. Review these terms thoroughly prior to any commercial application or integration.