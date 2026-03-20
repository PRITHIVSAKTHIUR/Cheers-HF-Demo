import gradio as gr
import numpy as np
import random
import torch
import spaces
import base64
import os
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MAX_SEED = np.iinfo(np.int32).max

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = "ai9stars/Cheers"
processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
model = model.to(dtype).to(device)
model.eval()

EXAMPLES_CONFIG = [
    {
        "images": ["examples/1.jpg"],
        "prompt": "Describe this image in detail.",
    },
]


def load_example_image_b64(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"
    except Exception:
        return ""


i2t_examples_html = ""
for idx, ex in enumerate(EXAMPLES_CONFIG):
    img_path = ex["images"][0] if ex.get("images") else ""
    b64 = load_example_image_b64(img_path)
    if b64:
        i2t_examples_html += (
            f'<div class="example-img-card" data-prompt="{ex["prompt"]}">'
            f'<img src="{b64}" alt="Example {idx + 1}"/>'
            f"<div>"
            f'<div class="example-img-label">Try this example</div>'
            f'<div class="example-img-prompt">{ex["prompt"]}</div>'
            f"</div></div>"
        )


def b64_to_pil(b64_str):
    if not b64_str or not b64_str.startswith("data:image"):
        return None
    try:
        _, data = b64_str.split(",", 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


@spaces.GPU
def run_inference(
    mode,
    b64_str,
    prompt,
    seed,
    randomize_seed,
    temperature,
    max_length,
    cfg_scale,
    num_inference_steps,
    alpha,
):
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter a prompt before processing.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    torch.manual_seed(seed)

    if mode == "Text-to-Image":
        images_batch = [None]
        messages_batch = [[{"role": "user", "content": prompt}]]
        texts = [
            processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages_batch
        ]
        inputs = processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            add_im_start_id=True,
        )
        inputs = {
            k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        gen_config = {
            "max_length": int(max_length),
            "cfg_scale": float(cfg_scale),
            "temperature": float(temperature),
            "num_inference_steps": int(num_inference_steps),
            "alpha": float(alpha),
            "edit_image": False,
        }
        inputs.update(gen_config)

        generated = model.generate(**inputs)
        images = generated["images"][0]
        current_img = images[0].clamp(0.0, 1.0)

        img_np = current_img.detach().permute(1, 2, 0).cpu().float().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        return pil_img, "", seed

    elif mode == "Image-to-Text":
        img = b64_to_pil(b64_str)
        if img is None:
            raise gr.Error("Please upload an image first.")

        content = f"<im_start><image><im_end>\n{prompt}"
        images_batch = [img]
        messages_batch = [[{"role": "user", "content": content}]]
        texts = [
            processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages_batch
        ]
        inputs = processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            add_im_start_id=False,
        )
        inputs = {
            k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        gen_config = {
            "max_length": int(max_length),
            "temperature": float(temperature),
        }
        inputs.update(gen_config)

        generated = model.generate(**inputs)
        input_ids = generated["input_ids"]
        result = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        text_result = result[0] if result else ""
        return None, text_result, seed

    else:
        images_batch = [None]
        messages_batch = [[{"role": "user", "content": prompt}]]
        texts = [
            processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages_batch
        ]
        inputs = processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            add_im_start_id=False,
        )
        inputs = {
            k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        gen_config = {
            "max_length": int(max_length),
            "temperature": float(temperature),
        }
        inputs.update(gen_config)

        generated = model.generate(**inputs)
        input_ids = generated["input_ids"]
        result = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        text_result = result[0] if result else ""
        return None, text_result, seed


css = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*{box-sizing:border-box;margin:0;padding:0}

body,.gradio-container{
    background:#0a0d0f!important;
    font-family:'Inter',system-ui,-apple-system,sans-serif!important;
    font-size:14px!important;
    color:#e4e4e7!important;
    min-height:100vh;
}
.dark body,.dark .gradio-container{
    background:#0a0d0f!important;
    color:#e4e4e7!important;
}
footer{display:none!important}

.hidden-input{
    display:none!important;
    height:0!important;
    overflow:hidden!important;
    margin:0!important;
    padding:0!important;
}

.app-shell{
    background:#13171a;
    border:1px solid #1e2529;
    border-radius:16px;
    margin:12px auto;
    max-width:1440px;
    overflow:hidden;
    box-shadow:0 25px 50px -12px rgba(0,0,0,.6),
               0 0 0 1px rgba(30,144,255,.04);
}

.app-header{
    background:linear-gradient(135deg,#13171a 0%,#181d21 100%);
    border-bottom:1px solid #1e2529;
    padding:14px 24px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:12px;
}
.app-header-left{
    display:flex;
    align-items:center;
    gap:12px;
}
.app-logo{
    width:36px;height:36px;
    background:linear-gradient(135deg,#1E90FF,#1873CC,#1260A8);
    border-radius:10px;
    display:flex;align-items:center;justify-content:center;
    box-shadow:0 4px 12px rgba(30,144,255,.3);
}
.app-logo svg{width:20px;height:20px}
.app-title{
    font-size:18px;font-weight:700;
    background:linear-gradient(135deg,#e4e4e7,#a1a1aa);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    letter-spacing:-.3px;
}
.app-badge{
    font-size:11px;font-weight:600;
    padding:3px 10px;border-radius:20px;
    background:rgba(30,144,255,.12);
    color:#63B3FF;
    border:1px solid rgba(30,144,255,.2);
    letter-spacing:.3px;
}

.mode-switcher{
    display:flex;gap:4px;
    background:#0a0d0f;
    border:1px solid #1e2529;
    border-radius:10px;padding:3px;
}
.mode-btn{
    display:inline-flex;align-items:center;justify-content:center;
    gap:6px;padding:6px 16px;border:none;border-radius:8px;
    cursor:pointer;font-size:13px;font-weight:600;
    font-family:'Inter',sans-serif;color:#6b7280;
    background:transparent;transition:all .2s ease;white-space:nowrap;
}
.mode-btn:hover{color:#a1a1aa;background:rgba(30,144,255,.06)}
.mode-btn svg{width:16px;height:16px;flex-shrink:0}
.mode-btn.active{
    color:#fff!important;-webkit-text-fill-color:#fff!important;
    background:linear-gradient(135deg,#1E90FF,#1873CC)!important;
    box-shadow:0 2px 8px rgba(30,144,255,.35);
}
.mode-btn.active svg,.mode-btn.active svg *{stroke:#fff!important;fill:none!important}
.mode-btn.disabled-mode{opacity:.5;cursor:not-allowed;pointer-events:none}

.app-main-row{display:flex;gap:0;flex:1;overflow:hidden;min-height:560px}
.app-main-left{flex:1;display:flex;flex-direction:column;min-width:0;border-right:1px solid #1e2529}
.app-main-right{width:440px;display:flex;flex-direction:column;flex-shrink:0;background:#13171a;overflow-y:auto}

.panel-card{border-bottom:1px solid #1e2529}
.panel-card-title{
    padding:12px 20px;font-size:12px;font-weight:600;color:#6b7280;
    text-transform:uppercase;letter-spacing:.8px;
    border-bottom:1px solid rgba(30,37,41,.6);
    display:flex;align-items:center;justify-content:space-between;
}
.panel-card-body{padding:16px 20px;display:flex;flex-direction:column;gap:8px}

.modern-textarea{
    width:100%;background:#0a0d0f;border:1px solid #1e2529;border-radius:8px;
    padding:12px 14px;font-family:'Inter',sans-serif;font-size:14px;color:#e4e4e7;
    resize:vertical;outline:none;min-height:120px;line-height:1.6;transition:border-color .2s;
}
.modern-textarea:focus{border-color:#1E90FF;box-shadow:0 0 0 3px rgba(30,144,255,.12)}
.modern-textarea::placeholder{color:#3f4854}

.example-chips{display:flex;flex-wrap:wrap;gap:6px;padding:8px 0 4px}
.example-chip{
    display:inline-flex;align-items:center;gap:4px;padding:5px 12px;
    background:rgba(30,144,255,.06);border:1px solid rgba(30,144,255,.15);
    border-radius:20px;cursor:pointer;font-size:12px;color:#A8D4FF;
    font-family:'Inter',sans-serif;font-weight:500;transition:all .15s;white-space:nowrap;
}
.example-chip:hover{background:rgba(30,144,255,.14);border-color:rgba(30,144,255,.3);color:#fff}

.example-img-card{
    display:flex;align-items:center;gap:12px;
    padding:10px 14px;
    background:rgba(30,144,255,.06);
    border:1px solid rgba(30,144,255,.15);
    border-radius:10px;cursor:pointer;
    transition:all .15s;width:100%;margin-top:4px;
}
.example-img-card:hover{
    background:rgba(30,144,255,.14);
    border-color:rgba(30,144,255,.3);
    transform:translateY(-1px);
    box-shadow:0 4px 12px rgba(0,0,0,.3);
}
.example-img-card img{
    width:52px;height:52px;border-radius:8px;
    object-fit:cover;border:1px solid #2a3038;flex-shrink:0;
}
.example-img-label{font-size:13px;color:#A8D4FF;font-weight:600}
.example-img-prompt{font-size:11px;color:#6b7280;margin-top:3px}

#image-upload-wrap{
    position:relative;background:#0a0d0f;min-height:280px;
    display:flex;align-items:center;justify-content:center;overflow:hidden;
}
.upload-prompt-modern{
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    width:100%;height:100%;min-height:280px;
}
.upload-click-area{
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    cursor:pointer;padding:36px 44px;border:2px dashed #2a3038;border-radius:16px;
    background:rgba(30,144,255,.02);transition:all .2s ease;gap:12px;
}
.upload-click-area:hover{background:rgba(30,144,255,.06);border-color:#1E90FF;transform:scale(1.02)}
.upload-click-area svg{width:64px;height:64px}
.upload-hint{font-size:13px;color:#6b7280;font-weight:500}
#image-preview{width:100%;text-align:center;padding:12px;position:relative}
#image-preview img{max-width:100%;max-height:320px;border-radius:8px;border:1px solid #1e2529}
.image-overlay-bar{display:flex;justify-content:center;gap:8px;padding:10px 0 4px}
.image-overlay-btn{
    display:inline-flex;align-items:center;gap:5px;padding:5px 14px;
    background:rgba(30,144,255,.1);border:1px solid rgba(30,144,255,.2);
    border-radius:6px;color:#A8D4FF;font-size:12px;font-weight:500;
    cursor:pointer;font-family:'Inter',sans-serif;transition:all .15s;
}
.image-overlay-btn:hover{background:rgba(30,144,255,.2);color:#fff}
.image-overlay-btn svg{width:13px;height:13px}

.hint-bar{
    background:rgba(30,144,255,.04);border-top:1px solid #1e2529;
    padding:14px 20px;font-size:13px;color:#6b7280;line-height:1.7;
    flex:1;display:flex;flex-direction:column;gap:6px;
}
.hint-bar b{color:#A8D4FF;font-weight:600}
.hint-bar kbd{
    display:inline-block;padding:1px 6px;background:#1e2529;
    border:1px solid #2a3038;border-radius:4px;
    font-family:'JetBrains Mono',monospace;font-size:11px;color:#a1a1aa;
}

.btn-run{
    display:flex;align-items:center;justify-content:center;gap:8px;
    width:100%;background:linear-gradient(135deg,#1E90FF,#1873CC);
    border:none;border-radius:10px;padding:13px 24px;cursor:pointer;
    font-size:15px;font-weight:700;font-family:'Inter',sans-serif;
    color:#fff!important;-webkit-text-fill-color:#fff!important;
    transition:all .2s ease;
    box-shadow:0 4px 16px rgba(30,144,255,.25),inset 0 1px 0 rgba(255,255,255,.15);
    letter-spacing:-.2px;
}
.btn-run:hover{
    background:linear-gradient(135deg,#63B3FF,#1E90FF);
    box-shadow:0 6px 24px rgba(30,144,255,.4),inset 0 1px 0 rgba(255,255,255,.2);
    transform:translateY(-1px);color:#fff!important;-webkit-text-fill-color:#fff!important;
}
.btn-run:active{transform:translateY(0);box-shadow:0 2px 8px rgba(30,144,255,.25)}
.btn-run:disabled{opacity:.6;cursor:not-allowed;transform:none!important}
.btn-run svg{width:18px;height:18px}
.btn-run svg,.btn-run svg *{fill:#fff!important;stroke:none!important}
.btn-run span,#run-btn-label{color:#fff!important;-webkit-text-fill-color:#fff!important}

.output-frame{border-bottom:1px solid #1e2529;display:flex;flex-direction:column;position:relative}
.output-frame .out-title{
    padding:10px 20px;font-size:12px;font-weight:700;color:#a1a1aa;
    text-transform:uppercase;letter-spacing:.8px;
    border-bottom:1px solid rgba(30,37,41,.6);
    display:flex;align-items:center;justify-content:space-between;
}
.output-frame .out-body{
    flex:1;background:#0a0d0f;display:flex;align-items:center;justify-content:center;
    overflow:hidden;min-height:220px;position:relative;
}
.output-frame .out-body img.modern-out-img{
    max-width:100%;max-height:460px;image-rendering:auto;animation:fadeInImage .5s ease;
}
@keyframes fadeInImage{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
.output-frame .out-placeholder{color:#2a3038;font-size:13px;text-align:center;padding:20px}
.out-action-btn{
    display:none;align-items:center;justify-content:center;
    background:rgba(30,144,255,.08);border:1px solid rgba(30,144,255,.18);
    border-radius:6px;cursor:pointer;padding:3px 10px;font-size:11px;font-weight:500;
    color:#A8D4FF!important;gap:4px;height:24px;transition:all .15s;font-family:'Inter',sans-serif;
}
.out-action-btn:hover{background:rgba(30,144,255,.18);border-color:rgba(30,144,255,.3);color:#fff!important}
.out-action-btn.visible{display:inline-flex}
.out-action-btn svg{width:12px;height:12px;fill:#A8D4FF}

.text-output-content{
    display:none;width:100%;padding:20px;font-family:'Inter',sans-serif;
    font-size:14px;line-height:1.75;color:#e4e4e7;white-space:pre-wrap;
    word-break:break-word;max-height:420px;overflow-y:auto;align-self:stretch;
    animation:fadeInText .3s ease;
}
@keyframes fadeInText{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.text-output-content::-webkit-scrollbar{width:8px}
.text-output-content::-webkit-scrollbar-track{background:#0a0d0f}
.text-output-content::-webkit-scrollbar-thumb{background:#1e2529;border-radius:4px}
.text-output-content::-webkit-scrollbar-thumb:hover{background:#2a3038}

.modern-loader{
    display:none;position:absolute;top:0;left:0;right:0;bottom:0;
    background:rgba(10,13,15,.92);z-index:15;flex-direction:column;
    align-items:center;justify-content:center;gap:14px;backdrop-filter:blur(4px);
}
.modern-loader.active{display:flex}
.modern-loader .loader-spinner{
    width:36px;height:36px;border:3px solid #1e2529;border-top-color:#1E90FF;
    border-radius:50%;animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.modern-loader .loader-text{font-size:13px;color:#6b7280;font-weight:500}
.loader-bar-track{width:200px;height:4px;background:#1e2529;border-radius:2px;overflow:hidden}
.loader-bar-fill{
    height:100%;background:linear-gradient(90deg,#1E90FF,#63B3FF,#1E90FF);
    background-size:200% 100%;animation:shimmer 1.5s ease-in-out infinite;
    border-radius:2px;width:100%;
}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

.toast-container{
    position:fixed;top:24px;right:24px;z-index:99999;
    display:flex;flex-direction:column;gap:10px;pointer-events:none;
}
.toast{
    pointer-events:auto;background:#1a1f24;border:1px solid #2a3038;
    border-left:4px solid #FF6B6B;border-radius:10px;padding:14px 20px;
    color:#e4e4e7;font-size:14px;font-family:'Inter',sans-serif;
    box-shadow:0 12px 40px rgba(0,0,0,.6);display:flex;align-items:center;
    gap:12px;animation:toastIn .35s ease;max-width:420px;min-width:280px;
}
.toast.error{border-left-color:#FF6B6B}
.toast.warning{border-left-color:#FFB347}
.toast.success{border-left-color:#1E90FF}
.toast.leaving{animation:toastOut .3s ease forwards}
@keyframes toastIn{from{transform:translateX(120%);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes toastOut{from{transform:translateX(0);opacity:1}to{transform:translateX(120%);opacity:0}}
.toast-icon{flex-shrink:0;display:flex;align-items:center}
.toast-message{flex:1;line-height:1.4}
.toast-close{
    flex-shrink:0;cursor:pointer;color:#6b7280;padding:2px;border:none;
    background:none;font-size:18px;line-height:1;font-family:'Inter',sans-serif;
    display:flex;align-items:center;
}
.toast-close:hover{color:#a1a1aa}

.settings-group{border:1px solid #1e2529;border-radius:10px;margin:12px 16px;overflow:hidden}
.settings-group-title{
    font-size:12px;font-weight:600;color:#6b7280;text-transform:uppercase;
    letter-spacing:.8px;padding:10px 16px;border-bottom:1px solid #1e2529;
    background:rgba(19,23,26,.5);display:flex;align-items:center;gap:8px;
}
.settings-group-title svg{width:14px;height:14px;flex-shrink:0}
.settings-group-body{padding:14px 16px;display:flex;flex-direction:column;gap:12px}
.slider-row{display:flex;align-items:center;gap:10px;min-height:28px}
.slider-row label{font-size:13px;font-weight:500;color:#a1a1aa;min-width:85px;flex-shrink:0}
.slider-row input[type="range"]{
    flex:1;-webkit-appearance:none;appearance:none;height:6px;
    background:#1e2529;border-radius:3px;outline:none;min-width:0;
}
.slider-row input[type="range"]::-webkit-slider-thumb{
    -webkit-appearance:none;appearance:none;width:16px;height:16px;
    background:linear-gradient(135deg,#1E90FF,#1873CC);border-radius:50%;
    cursor:pointer;box-shadow:0 2px 6px rgba(30,144,255,.35);transition:transform .15s;
}
.slider-row input[type="range"]::-webkit-slider-thumb:hover{transform:scale(1.2)}
.slider-row input[type="range"]::-moz-range-thumb{
    width:16px;height:16px;background:linear-gradient(135deg,#1E90FF,#1873CC);
    border-radius:50%;cursor:pointer;border:none;box-shadow:0 2px 6px rgba(30,144,255,.35);
}
.slider-row .slider-val{
    min-width:52px;text-align:right;font-family:'JetBrains Mono',monospace;
    font-size:12px;font-weight:500;padding:3px 8px;background:#0a0d0f;
    border:1px solid #1e2529;border-radius:6px;color:#a1a1aa;flex-shrink:0;
}
.checkbox-row{display:flex;align-items:center;gap:8px;font-size:13px;color:#a1a1aa}
.checkbox-row input[type="checkbox"]{accent-color:#1E90FF;width:16px;height:16px;cursor:pointer}
.checkbox-row label{color:#a1a1aa;font-size:13px;cursor:pointer}
.settings-divider{height:1px;background:#1e2529;margin:4px 0}

.app-statusbar{
    background:#13171a;border-top:1px solid #1e2529;padding:6px 20px;
    display:flex;gap:12px;height:34px;align-items:center;font-size:12px;
}
.app-statusbar .sb-section{
    padding:0 12px;flex:1;display:flex;align-items:center;
    font-family:'JetBrains Mono',monospace;font-size:12px;color:#3f4854;
    overflow:hidden;white-space:nowrap;
}
.app-statusbar .sb-mode{
    flex:0 0 auto;min-width:130px;text-align:center;justify-content:center;
    padding:3px 12px;background:rgba(30,144,255,.06);border-radius:6px;
    color:#63B3FF;font-weight:500;font-family:'JetBrains Mono',monospace;font-size:12px;
}
.app-statusbar .sb-fixed{
    flex:0 0 auto;min-width:90px;text-align:center;justify-content:center;
    padding:3px 12px;background:rgba(30,144,255,.06);border-radius:6px;
    color:#63B3FF;font-weight:500;
}

#gradio-run-btn{
    position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;
    opacity:0.01;pointer-events:none;overflow:hidden;
}

::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:#0a0d0f}
::-webkit-scrollbar-thumb{background:#1e2529;border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:#2a3038}

.dark .app-shell{background:#13171a}
.dark .btn-run,.dark .btn-run *,.dark .btn-run span,.dark #run-btn-label{
    color:#fff!important;-webkit-text-fill-color:#fff!important;
}
.dark .btn-run svg,.dark .btn-run svg *{fill:#fff!important}
.dark .mode-btn.active,.dark .mode-btn.active *{color:#fff!important;-webkit-text-fill-color:#fff!important}
body:not(.dark) .btn-run,body:not(.dark) .btn-run *,
body:not(.dark) .btn-run span,body:not(.dark) #run-btn-label{
    color:#fff!important;-webkit-text-fill-color:#fff!important;
}
body:not(.dark) .mode-btn.active,body:not(.dark) .mode-btn.active *{
    color:#fff!important;-webkit-text-fill-color:#fff!important;
}
.gradio-container .btn-run,.gradio-container .btn-run *,
.gradio-container #run-btn-label{color:#fff!important;-webkit-text-fill-color:#fff!important}

@media(max-width:860px){
    .app-main-row{flex-direction:column}
    .app-main-right{width:100%}
    .app-main-left{border-right:none;border-bottom:1px solid #1e2529}
    .mode-switcher{flex-wrap:wrap}
    .app-header{flex-direction:column;align-items:flex-start}
}
"""

init_js = r"""
() => {
function initCheers() {
    if (window.__cheersInitDone) return;

    var modeT2I = document.getElementById('mode-t2i');
    var modeI2T = document.getElementById('mode-i2t');
    var modeT2T = document.getElementById('mode-t2t');

    var imageUploadPanel = document.getElementById('image-upload-panel');
    var outputImageSection = document.getElementById('output-image-section');
    var outputTextSection = document.getElementById('output-text-section');
    var t2iSettings = document.getElementById('t2i-settings');
    var hintBar = document.getElementById('hint-bar');
    var runBtnLabel = document.getElementById('run-btn-label');
    var sbMode = document.getElementById('sb-mode-label');
    var sbStatus = document.getElementById('sb-status');
    var outputTextTitle = document.getElementById('output-text-title');
    var promptInput = document.getElementById('custom-prompt-input');

    var uploadPrompt = document.getElementById('upload-prompt');
    var uploadClickArea = document.getElementById('upload-click-area');
    var imagePreview = document.getElementById('image-preview');
    var previewImg = document.getElementById('preview-img');
    var fileInput = document.getElementById('custom-file-input');
    var btnChangeImage = document.getElementById('btn-change-image');
    var btnRemoveImage = document.getElementById('btn-remove-image');
    var imageUploadWrap = document.getElementById('image-upload-wrap');

    var customRunBtn = document.getElementById('custom-run-btn');

    var chipsT2I = document.getElementById('example-chips-t2i');
    var chipsI2T = document.getElementById('example-chips-i2t');
    var chipsT2T = document.getElementById('example-chips-t2t');
    var i2tImgExamples = document.getElementById('i2t-image-examples');

    if (!modeT2I || !promptInput || !customRunBtn) {
        setTimeout(initCheers, 250);
        return;
    }

    window.__cheersInitDone = true;
    window.__currentMode = 'Text-to-Image';
    window.__isProcessing = false;
    window.__outputReceived = false;

    var DEFAULT_PROMPTS = {
        'Text-to-Image': 'A serene mountain landscape with a crystal clear lake reflecting snow-capped peaks at golden hour, photorealistic',
        'Image-to-Text': 'Describe this image in detail.',
        'Text-to-Text': 'Please introduce yourself briefly.'
    };

    var HINTS = {
        'Text-to-Image': '<b>Text to Image</b> — Enter a detailed description of the image you want to generate. More detail yields better results. Adjust <kbd>CFG Scale</kbd> for prompt adherence and <kbd>Steps</kbd> for quality.',
        'Image-to-Text': '<b>Image to Text</b> — Upload an image and enter a question or instruction. The model will analyze the image and generate a text response.',
        'Text-to-Text': '<b>Text to Text</b> — Type your message or question. The model will generate a text response. Adjust <kbd>Temperature</kbd> for creativity.'
    };

    function showToast(message, type) {
        type = type || 'error';
        var container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        var iconSvg = '';
        if (type === 'error') {
            iconSvg = '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#FF6B6B" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>';
        } else if (type === 'warning') {
            iconSvg = '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#FFB347" stroke-width="2"><path d="M12 2L2 22h20L12 2z"/><path d="M12 10v4M12 18h.01"/></svg>';
        } else {
            iconSvg = '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#1E90FF" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 12l3 3 5-5"/></svg>';
        }
        var toast = document.createElement('div');
        toast.className = 'toast ' + type;
        toast.innerHTML = '<span class="toast-icon">' + iconSvg + '</span><span class="toast-message">' + message + '</span><button class="toast-close" onclick="this.parentElement.classList.add(\'leaving\');setTimeout(function(){toast.remove()},300)">&times;</button>';
        container.appendChild(toast);
        var ref = toast;
        setTimeout(function() {
            ref.classList.add('leaving');
            setTimeout(function() { ref.remove(); }, 300);
        }, 4000);
    }
    window.__showToast = showToast;

    var failsafeTimer = null;

    function lockModeSwitcher() {
        modeT2I.classList.add('disabled-mode');
        modeI2T.classList.add('disabled-mode');
        modeT2T.classList.add('disabled-mode');
    }
    function unlockModeSwitcher() {
        modeT2I.classList.remove('disabled-mode');
        modeI2T.classList.remove('disabled-mode');
        modeT2T.classList.remove('disabled-mode');
    }

    function showLoaders() {
        window.__isProcessing = true;
        window.__outputReceived = false;
        var ids = ['output-loader', 'text-loader'];
        for (var i = 0; i < ids.length; i++) {
            var l = document.getElementById(ids[i]);
            if (l) l.classList.add('active');
        }
        customRunBtn.disabled = true;
        lockModeSwitcher();
        if (sbStatus) sbStatus.textContent = 'Processing...';
        if (failsafeTimer) clearTimeout(failsafeTimer);
        failsafeTimer = setTimeout(function() {
            if (window.__isProcessing && !window.__outputReceived) {
                hideLoaders();
                showToast('Processing timed out. Please try again.', 'error');
            }
        }, 600000);
    }

    function hideLoaders() {
        if (!window.__isProcessing) return;
        window.__isProcessing = false;
        window.__outputReceived = true;
        if (failsafeTimer) { clearTimeout(failsafeTimer); failsafeTimer = null; }
        var ids = ['output-loader', 'text-loader'];
        for (var i = 0; i < ids.length; i++) {
            var l = document.getElementById(ids[i]);
            if (l) l.classList.remove('active');
        }
        customRunBtn.disabled = false;
        unlockModeSwitcher();
        if (sbStatus) sbStatus.textContent = 'Done';
    }
    window.__showLoaders = showLoaders;
    window.__hideLoaders = hideLoaders;

    function setGradioValue(containerId, value) {
        var container = document.getElementById(containerId);
        if (!container) return;
        var allInputs = container.querySelectorAll('input, textarea');
        allInputs.forEach(function(el) {
            if (el.type === 'file' || el.type === 'range' || el.type === 'checkbox') return;
            var proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            var ns = Object.getOwnPropertyDescriptor(proto, 'value');
            if (ns && ns.set) {
                ns.set.call(el, value);
                el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            }
        });
    }

    function setSliderValue(customId, value) {
        var slider = document.getElementById(customId);
        var valSpan = document.getElementById(customId + '-val');
        if (slider) {
            slider.value = value;
            if (valSpan) valSpan.textContent = value;
        }
    }

    function switchMode(mode) {
        if (window.__isProcessing) {
            showToast('Please wait for the current process to complete.', 'warning');
            return;
        }
        window.__currentMode = mode;

        modeT2I.classList.toggle('active', mode === 'Text-to-Image');
        modeI2T.classList.toggle('active', mode === 'Image-to-Text');
        modeT2T.classList.toggle('active', mode === 'Text-to-Text');

        if (chipsT2I) chipsT2I.style.display = mode === 'Text-to-Image' ? 'flex' : 'none';
        if (chipsI2T) chipsI2T.style.display = mode === 'Image-to-Text' ? 'flex' : 'none';
        if (chipsT2T) chipsT2T.style.display = mode === 'Text-to-Text' ? 'flex' : 'none';
        if (i2tImgExamples) i2tImgExamples.style.display = mode === 'Image-to-Text' ? 'block' : 'none';

        if (mode === 'Text-to-Image') {
            imageUploadPanel.style.display = 'none';
            outputImageSection.style.display = '';
            outputTextSection.style.display = 'none';
            t2iSettings.style.display = '';
            promptInput.placeholder = 'Describe the image you want to generate...';
            promptInput.style.minHeight = '180px';
            runBtnLabel.textContent = 'Generate Image';
            sbMode.textContent = 'Text to Image';
            setSliderValue('custom-temperature', '0.0');
            setSliderValue('custom-max-length', '300');
        } else if (mode === 'Image-to-Text') {
            imageUploadPanel.style.display = '';
            outputImageSection.style.display = 'none';
            outputTextSection.style.display = '';
            t2iSettings.style.display = 'none';
            promptInput.placeholder = 'Ask a question about the image...';
            promptInput.style.minHeight = '80px';
            runBtnLabel.textContent = 'Analyze Image';
            sbMode.textContent = 'Image to Text';
            outputTextTitle.textContent = 'Image Description';
            setSliderValue('custom-temperature', '0.3');
            setSliderValue('custom-max-length', '150');
        } else {
            imageUploadPanel.style.display = 'none';
            outputImageSection.style.display = 'none';
            outputTextSection.style.display = '';
            t2iSettings.style.display = 'none';
            promptInput.placeholder = 'Type your message or question...';
            promptInput.style.minHeight = '180px';
            runBtnLabel.textContent = 'Generate Response';
            sbMode.textContent = 'Text to Text';
            outputTextTitle.textContent = 'Response';
            setSliderValue('custom-temperature', '0.3');
            setSliderValue('custom-max-length', '100');
        }

        promptInput.value = DEFAULT_PROMPTS[mode];
        hintBar.innerHTML = HINTS[mode];
        setGradioValue('mode-gradio-input', mode);
        syncAllToGradio();
    }

    modeT2I.addEventListener('click', function() { switchMode('Text-to-Image'); });
    modeI2T.addEventListener('click', function() { switchMode('Image-to-Text'); });
    modeT2T.addEventListener('click', function() { switchMode('Text-to-Text'); });

    function processFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            showToast('Please select a valid image file.', 'warning');
            return;
        }
        var reader = new FileReader();
        reader.onload = function(event) {
            var dataUrl = event.target.result;
            previewImg.src = dataUrl;
            uploadPrompt.style.display = 'none';
            imagePreview.style.display = 'block';
            setGradioValue('hidden-image-b64', dataUrl);
        };
        reader.onerror = function() {
            showToast('Failed to read the image file.', 'error');
        };
        reader.readAsDataURL(file);
    }

    function openFilePicker() { fileInput.click(); }

    if (uploadClickArea) uploadClickArea.addEventListener('click', openFilePicker);
    if (btnChangeImage) btnChangeImage.addEventListener('click', openFilePicker);
    if (btnRemoveImage) {
        btnRemoveImage.addEventListener('click', function() {
            previewImg.src = '';
            imagePreview.style.display = 'none';
            uploadPrompt.style.display = 'flex';
            setGradioValue('hidden-image-b64', '');
        });
    }
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (e.target.files && e.target.files[0]) {
                processFile(e.target.files[0]);
            }
            e.target.value = '';
        });
    }

    if (imageUploadWrap) {
        imageUploadWrap.addEventListener('dragover', function(e) {
            e.preventDefault(); e.stopPropagation();
            imageUploadWrap.style.outline = '2px solid #1E90FF';
            imageUploadWrap.style.outlineOffset = '-2px';
        });
        imageUploadWrap.addEventListener('dragleave', function(e) {
            e.preventDefault(); e.stopPropagation();
            imageUploadWrap.style.outline = '';
        });
        imageUploadWrap.addEventListener('drop', function(e) {
            e.preventDefault(); e.stopPropagation();
            imageUploadWrap.style.outline = '';
            if (e.dataTransfer.files && e.dataTransfer.files.length) {
                processFile(e.dataTransfer.files[0]);
            }
        });
    }

    var allChips = document.querySelectorAll('.example-chip');
    allChips.forEach(function(chip) {
        chip.addEventListener('click', function() {
            if (window.__isProcessing) return;
            var text = chip.getAttribute('data-prompt');
            if (text && promptInput) {
                promptInput.value = text;
                syncAllToGradio();
            }
        });
    });

    var imgCards = document.querySelectorAll('.example-img-card');
    imgCards.forEach(function(card) {
        card.addEventListener('click', function() {
            if (window.__isProcessing) return;
            var prompt = card.getAttribute('data-prompt');
            var imgEl = card.querySelector('img');
            var imageData = imgEl ? imgEl.src : '';
            if (prompt && promptInput) {
                promptInput.value = prompt;
            }
            if (imageData && previewImg) {
                previewImg.src = imageData;
                uploadPrompt.style.display = 'none';
                imagePreview.style.display = 'block';
                setGradioValue('hidden-image-b64', imageData);
            }
            syncAllToGradio();
            showToast('Example loaded! Click Analyze Image to process.', 'success');
        });
    });

    function syncAllToGradio() {
        setGradioValue('prompt-gradio-input', promptInput.value);
        setGradioValue('mode-gradio-input', window.__currentMode);
        var sliders = [
            ['custom-seed', 'gradio-seed'],
            ['custom-temperature', 'gradio-temperature'],
            ['custom-max-length', 'gradio-max-length'],
            ['custom-cfg-scale', 'gradio-cfg-scale'],
            ['custom-steps', 'gradio-steps'],
            ['custom-alpha', 'gradio-alpha']
        ];
        sliders.forEach(function(pair) {
            var cid = pair[0], gid = pair[1];
            var el = document.getElementById(cid);
            if (!el) return;
            var container = document.getElementById(gid);
            if (!container) return;
            var targets = [];
            container.querySelectorAll('input[type="range"]').forEach(function(t){targets.push(t)});
            container.querySelectorAll('input[type="number"]').forEach(function(t){targets.push(t)});
            targets.forEach(function(t) {
                var ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                if (ns && ns.set) {
                    ns.set.call(t, el.value);
                    t.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                    t.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                }
            });
        });
        var randCheck = document.getElementById('custom-randomize');
        if (randCheck) {
            var container = document.getElementById('gradio-randomize');
            if (container) {
                var cb = container.querySelector('input[type="checkbox"]');
                if (cb && cb.checked !== randCheck.checked) cb.click();
            }
        }
    }

    function syncSlider(customId, gradioId) {
        var slider = document.getElementById(customId);
        var valSpan = document.getElementById(customId + '-val');
        if (!slider) return;
        slider.addEventListener('input', function() {
            if (valSpan) valSpan.textContent = slider.value;
            var container = document.getElementById(gradioId);
            if (!container) return;
            var targets = [];
            container.querySelectorAll('input[type="range"]').forEach(function(t){targets.push(t)});
            container.querySelectorAll('input[type="number"]').forEach(function(t){targets.push(t)});
            targets.forEach(function(el) {
                var ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                if (ns && ns.set) {
                    ns.set.call(el, slider.value);
                    el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                    el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                }
            });
        });
    }
    syncSlider('custom-seed', 'gradio-seed');
    syncSlider('custom-temperature', 'gradio-temperature');
    syncSlider('custom-max-length', 'gradio-max-length');
    syncSlider('custom-cfg-scale', 'gradio-cfg-scale');
    syncSlider('custom-steps', 'gradio-steps');
    syncSlider('custom-alpha', 'gradio-alpha');

    var randCheck = document.getElementById('custom-randomize');
    if (randCheck) {
        randCheck.addEventListener('change', function() {
            var container = document.getElementById('gradio-randomize');
            if (!container) return;
            var cb = container.querySelector('input[type="checkbox"]');
            if (cb && cb.checked !== randCheck.checked) cb.click();
        });
    }

    if (promptInput) {
        promptInput.addEventListener('input', function() {
            setGradioValue('prompt-gradio-input', promptInput.value);
        });
    }

    window.__clickGradioRunBtn = function() {
        syncAllToGradio();
        showLoaders();
        setTimeout(function() {
            var gradioBtn = document.getElementById('gradio-run-btn');
            if (!gradioBtn) return;
            var btn = gradioBtn.querySelector('button');
            if (btn) btn.click();
            else gradioBtn.click();
        }, 200);
    };

    customRunBtn.addEventListener('click', function() {
        if (window.__isProcessing) {
            showToast('Please wait for the current process to complete.', 'warning');
            return;
        }
        var prompt = promptInput.value.trim();
        if (!prompt) {
            showToast('Please enter a prompt before generating.', 'error');
            return;
        }
        if (window.__currentMode === 'Image-to-Text') {
            if (!previewImg.src || previewImg.src === '' || previewImg.src === window.location.href) {
                showToast('Please upload an image first.', 'error');
                return;
            }
        }
        window.__clickGradioRunBtn();
    });

    switchMode('Text-to-Image');
}
initCheers();
}
"""

wire_outputs_js = r"""
() => {
function downloadImage(imgSrc, filename) {
    var a = document.createElement('a');
    a.href = imgSrc;
    a.download = filename || 'image.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text);
    } else {
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    }
}

function watchOutputs() {
    var resultImgContainer = document.getElementById('gradio-result-image');
    var resultTxtContainer = document.getElementById('gradio-result-text');
    var outImgBody = document.getElementById('output-image-container');
    var outTxtBody = document.getElementById('output-text-container');
    var outImgPh = document.getElementById('output-img-placeholder');
    var outTxtPh = document.getElementById('output-text-placeholder');
    var outTxtContent = document.getElementById('output-text-content');
    var dlBtnOut = document.getElementById('dl-btn-output');
    var copyBtnOut = document.getElementById('copy-btn-output');
    var sbSeed = document.getElementById('sb-seed');

    if (!resultImgContainer || !resultTxtContainer || !outImgBody || !outTxtBody) {
        setTimeout(watchOutputs, 500);
        return;
    }

    if (dlBtnOut) {
        dlBtnOut.addEventListener('click', function(e) {
            e.stopPropagation();
            var img = outImgBody.querySelector('img.modern-out-img');
            if (img && img.src) downloadImage(img.src, 'cheers_output.png');
        });
    }
    if (copyBtnOut) {
        copyBtnOut.addEventListener('click', function(e) {
            e.stopPropagation();
            if (outTxtContent && outTxtContent.textContent) {
                copyToClipboard(outTxtContent.textContent);
                copyBtnOut.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg> Copied!';
                setTimeout(function() {
                    copyBtnOut.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg> Copy';
                }, 2000);
            }
        });
    }

    var lastImgSrc = '';
    var lastTxtVal = '';

    function syncOutputs() {
        var resultImg = resultImgContainer.querySelector('img');
        if (resultImg && resultImg.src && resultImg.src !== lastImgSrc && !resultImg.src.endsWith('#')) {
            lastImgSrc = resultImg.src;
            if (outImgPh) outImgPh.style.display = 'none';
            var existing = outImgBody.querySelector('img.modern-out-img');
            if (!existing) {
                existing = document.createElement('img');
                existing.className = 'modern-out-img';
                outImgBody.appendChild(existing);
            }
            existing.src = resultImg.src;
            if (dlBtnOut) dlBtnOut.classList.add('visible');
            if (window.__isProcessing) {
                window.__outputReceived = true;
                if (window.__hideLoaders) window.__hideLoaders();
            }
        }

        var txtEl = resultTxtContainer.querySelector('textarea') || resultTxtContainer.querySelector('input[type="text"]');
        if (txtEl && txtEl.value && txtEl.value !== lastTxtVal) {
            lastTxtVal = txtEl.value;
            if (outTxtPh) outTxtPh.style.display = 'none';
            if (outTxtContent) {
                outTxtContent.textContent = txtEl.value;
                outTxtContent.style.display = 'block';
            }
            if (copyBtnOut) copyBtnOut.classList.add('visible');
            if (window.__isProcessing) {
                window.__outputReceived = true;
                if (window.__hideLoaders) window.__hideLoaders();
            }
        }

        var seedContainer = document.getElementById('gradio-seed');
        if (seedContainer && sbSeed) {
            var seedInput = seedContainer.querySelector('input[type="number"]') || seedContainer.querySelector('input[type="range"]');
            if (seedInput) {
                var sv = document.getElementById('custom-seed-val');
                var ss = document.getElementById('custom-seed');
                if (sv) sv.textContent = seedInput.value;
                if (ss) ss.value = seedInput.value;
                sbSeed.textContent = 'Seed: ' + seedInput.value;
            }
        }
    }

    var observer = new MutationObserver(syncOutputs);
    observer.observe(resultImgContainer, {childList:true, subtree:true, attributes:true, attributeFilter:['src']});
    observer.observe(resultTxtContainer, {childList:true, subtree:true, attributes:true, characterData:true});
    setInterval(syncOutputs, 500);
}
watchOutputs();
}
"""

LOGO_SVG = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M4 3h16l-1 4H5L4 3z" fill="none" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/><path d="M5 7v7c0 3.31 2.69 6 6 6h2c3.31 0 6-2.69 6-6V7" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M19 8c1.5.5 3 1.5 3 4s-1.5 3.5-3 4" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M8 1v3M12 1v3M16 1v3" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round"/></svg>'
T2I_SVG = '<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="3.5" width="14" height="11" rx="2"/><circle cx="6.5" cy="7.5" r="1.5"/><path d="M2 12l3.5-3 2.5 2 4-4 4 3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
I2T_SVG = '<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1.5 9s3-5.5 7.5-5.5S16.5 9 16.5 9s-3 5.5-7.5 5.5S1.5 9 1.5 9z"/><circle cx="9" cy="9" r="2.5"/></svg>'
T2T_SVG = '<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 3.5h12a1.5 1.5 0 011.5 1.5v7a1.5 1.5 0 01-1.5 1.5H7L3.5 17V13.5H3A1.5 1.5 0 011.5 12V5A1.5 1.5 0 013 3.5z"/><path d="M5.5 7.5h7M5.5 10h4" stroke-linecap="round"/></svg>'
UPLOAD_SVG = '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 42V20M32 20L22 30M32 20L42 30" stroke="#1E90FF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M46 40c5.5 0 8-4 8-8s-3.5-9-8.5-9c-.8 0-1.5.15-2.2.4C41.5 17.5 37 14 31 14c-7 0-12 5.5-12 12.5 0 .4 0 .8.08 1.2C15.5 29 13 32.5 13 36c0 5 3.8 9 8.5 9" stroke="#1E90FF" stroke-width="2.5" stroke-linecap="round"/></svg>'
RUN_SVG = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M19 7l-7 5-7-5V5l7 5 7-5v2zm0 6l-7 5-7-5v-2l7 5 7-5v2z"/></svg>'
DOWNLOAD_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 16l-5-5h3V4h4v7h3l-5 5z" fill="currentColor" stroke="none"/><path d="M20 18H4v2h16v-2z" fill="currentColor" stroke="none"/></svg>'
COPY_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>'
SETTINGS_SVG = '<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="9" cy="9" r="2.5"/><path d="M9 1.5v2M9 14.5v2M1.5 9h2M14.5 9h2M3.4 3.4l1.4 1.4M13.2 13.2l1.4 1.4M3.4 14.6l1.4-1.4M13.2 4.8l1.4-1.4"/></svg>'
CHANGE_SVG = '<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14.5 2.5l1 1-9.5 9.5H4v-2L14.5 2.5z"/><path d="M12 5l2 2" stroke-linecap="round"/></svg>'
REMOVE_SVG = '<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4.5 4.5l9 9M13.5 4.5l-9 9" stroke-linecap="round"/></svg>'

with gr.Blocks() as demo:

    hidden_image_b64 = gr.Textbox(elem_id="hidden-image-b64", elem_classes="hidden-input", container=False)
    mode_input = gr.Textbox(value="Text-to-Image", elem_id="mode-gradio-input", elem_classes="hidden-input", container=False)
    prompt_input = gr.Textbox(
        value="A serene mountain landscape with a crystal clear lake reflecting snow-capped peaks at golden hour, photorealistic",
        elem_id="prompt-gradio-input", elem_classes="hidden-input", container=False,
    )
    seed = gr.Slider(minimum=0, maximum=MAX_SEED, step=1, value=0, elem_id="gradio-seed", elem_classes="hidden-input", container=False)
    randomize_seed = gr.Checkbox(value=True, elem_id="gradio-randomize", elem_classes="hidden-input", container=False)
    temperature = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.0, elem_id="gradio-temperature", elem_classes="hidden-input", container=False)
    max_length = gr.Slider(minimum=50, maximum=1000, step=10, value=300, elem_id="gradio-max-length", elem_classes="hidden-input", container=False)
    cfg_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=9.5, elem_id="gradio-cfg-scale", elem_classes="hidden-input", container=False)
    num_inference_steps = gr.Slider(minimum=10, maximum=150, step=5, value=80, elem_id="gradio-steps", elem_classes="hidden-input", container=False)
    alpha = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, elem_id="gradio-alpha", elem_classes="hidden-input", container=False)
    result_image = gr.Image(elem_id="gradio-result-image", elem_classes="hidden-input", container=False, format="png")
    result_text = gr.Textbox(elem_id="gradio-result-text", elem_classes="hidden-input", container=False)

    gr.HTML(f"""
    <div class="app-shell">

        <div class="app-header">
            <div class="app-header-left">
                <div class="app-logo">{LOGO_SVG}</div>
                <span class="app-title">Cheers</span>
                <span class="app-badge">Unified</span>
            </div>
            <div class="mode-switcher">
                <button id="mode-t2i" class="mode-btn active" title="Text to Image">{T2I_SVG} Text-to-Image</button>
                <button id="mode-i2t" class="mode-btn" title="Image to Text">{I2T_SVG} Image-Text-to-Text</button>
                <button id="mode-t2t" class="mode-btn" title="Text to Text">{T2T_SVG} Text-to-Text</button>
            </div>
        </div>

        <div class="app-main-row">
            <div class="app-main-left">

                <div class="panel-card" id="prompt-panel">
                    <div class="panel-card-title">Input Prompt</div>
                    <div class="panel-card-body">
                        <textarea id="custom-prompt-input" class="modern-textarea" rows="6" style="min-height:180px"
                            placeholder="Describe the image you want to generate...">A serene mountain landscape with a crystal clear lake reflecting snow-capped peaks at golden hour, photorealistic</textarea>

                        <div class="example-chips" id="example-chips-t2i">
                            <span class="example-chip" data-prompt="A majestic wolf standing on a mountain peak under the northern lights, digital art">Wolf &amp; Aurora</span>
                            <span class="example-chip" data-prompt="Futuristic cyberpunk city with flying cars and neon signs reflecting in rain-soaked streets at night">Cyberpunk City</span>
                            <span class="example-chip" data-prompt="A cozy cabin in a snowy forest with warm light glowing from the windows, oil painting style">Cozy Cabin</span>
                            <span class="example-chip" data-prompt="An underwater scene with bioluminescent jellyfish and coral reefs, ethereal lighting">Underwater</span>
                            <span class="example-chip" data-prompt="Portrait of an astronaut floating in deep space with Earth reflected in the helmet visor, cinematic">Astronaut</span>
                            <span class="example-chip" data-prompt="A magical treehouse village connected by rope bridges in an enchanted forest, fantasy art">Treehouse Village</span>
                            <span class="example-chip" data-prompt="A steampunk airship flying over Victorian London at sunset, detailed illustration">Steampunk Airship</span>
                            <span class="example-chip" data-prompt="Japanese zen garden with cherry blossoms falling over a koi pond, watercolor style">Zen Garden</span>
                            <span class="example-chip" data-prompt="A dragon perched on a castle tower during a thunderstorm, epic fantasy, dramatic lighting">Dragon Castle</span>
                            <span class="example-chip" data-prompt="Retro 80s style sunset with palm trees and a DeLorean on a neon grid road, synthwave aesthetic">Synthwave Sunset</span>
                        </div>

                        <div class="example-chips" id="example-chips-i2t" style="display:none">
                            <span class="example-chip" data-prompt="Describe this image in detail.">Describe</span>
                            <span class="example-chip" data-prompt="What objects are in this image?">Objects</span>
                            <span class="example-chip" data-prompt="What is the mood or atmosphere of this image?">Mood</span>
                            <span class="example-chip" data-prompt="Generate a creative caption for this image.">Caption</span>
                        </div>

                        <div id="i2t-image-examples" style="display:none">
                            {i2t_examples_html}
                        </div>

                        <div class="example-chips" id="example-chips-t2t" style="display:none">
                            <span class="example-chip" data-prompt="Write a short poem about the ocean at sunset.">Poem</span>
                            <span class="example-chip" data-prompt="Explain quantum computing in simple terms.">Explain</span>
                            <span class="example-chip" data-prompt="Tell me an interesting fact about space.">Fun Fact</span>
                            <span class="example-chip" data-prompt="Write a haiku about artificial intelligence.">Haiku</span>
                        </div>
                    </div>
                </div>

                <div class="panel-card" id="image-upload-panel" style="display:none;">
                    <div class="panel-card-title">
                        Input Image
                        <span id="btn-change-image" class="image-overlay-btn" style="cursor:pointer">{CHANGE_SVG} Change</span>
                    </div>
                    <div class="panel-card-body" style="padding:0;">
                        <div id="image-upload-wrap">
                            <div id="upload-prompt" class="upload-prompt-modern">
                                <div id="upload-click-area" class="upload-click-area">
                                    {UPLOAD_SVG}
                                    <span class="upload-hint">Click or drag an image here</span>
                                </div>
                            </div>
                            <div id="image-preview" style="display:none;">
                                <img id="preview-img" src="" alt="Preview"/>
                                <div class="image-overlay-bar">
                                    <span id="btn-remove-image" class="image-overlay-btn" style="cursor:pointer">{REMOVE_SVG} Remove</span>
                                </div>
                            </div>
                        </div>
                        <input id="custom-file-input" type="file" accept="image/*" style="display:none;"/>
                    </div>
                </div>

                <div class="hint-bar" id="hint-bar">
                    <b>Text to Image</b> — Enter a detailed description of the image you want to generate. More detail yields better results.
                    Adjust <kbd>CFG Scale</kbd> for prompt adherence and <kbd>Steps</kbd> for quality.
                </div>
            </div>

            <div class="app-main-right">
                <div style="padding:14px 20px;">
                    <button id="custom-run-btn" class="btn-run">{RUN_SVG}<span id="run-btn-label">Generate Image</span></button>
                </div>

                <div id="output-image-section" class="output-frame" style="flex:1">
                    <div class="out-title">
                        <span>Generated Image</span>
                        <span id="dl-btn-output" class="out-action-btn" title="Download">{DOWNLOAD_SVG} Save</span>
                    </div>
                    <div class="out-body" id="output-image-container">
                        <div class="modern-loader" id="output-loader">
                            <div class="loader-spinner"></div>
                            <div class="loader-text">Generating image...</div>
                            <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
                        </div>
                        <div class="out-placeholder" id="output-img-placeholder">Generated image will appear here</div>
                    </div>
                </div>

                <div id="output-text-section" class="output-frame" style="display:none;flex:1">
                    <div class="out-title">
                        <span id="output-text-title">Response</span>
                        <span id="copy-btn-output" class="out-action-btn" title="Copy to clipboard">{COPY_SVG} Copy</span>
                    </div>
                    <div class="out-body" id="output-text-container" style="align-items:flex-start;">
                        <div class="modern-loader" id="text-loader">
                            <div class="loader-spinner"></div>
                            <div class="loader-text">Generating response...</div>
                            <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
                        </div>
                        <div class="out-placeholder" id="output-text-placeholder">Response will appear here</div>
                        <div id="output-text-content" class="text-output-content"></div>
                    </div>
                </div>

                <div class="settings-group">
                    <div class="settings-group-title">{SETTINGS_SVG} Advanced Settings</div>
                    <div class="settings-group-body">
                        <div class="slider-row">
                            <label>Seed</label>
                            <input type="range" id="custom-seed" min="0" max="2147483647" step="1" value="0">
                            <span class="slider-val" id="custom-seed-val">0</span>
                        </div>
                        <div class="checkbox-row">
                            <input type="checkbox" id="custom-randomize" checked>
                            <label for="custom-randomize">Randomize seed</label>
                        </div>
                        <div class="settings-divider"></div>
                        <div class="slider-row">
                            <label>Temperature</label>
                            <input type="range" id="custom-temperature" min="0" max="2" step="0.1" value="0.0">
                            <span class="slider-val" id="custom-temperature-val">0.0</span>
                        </div>
                        <div class="slider-row">
                            <label>Max Length</label>
                            <input type="range" id="custom-max-length" min="50" max="1000" step="10" value="300">
                            <span class="slider-val" id="custom-max-length-val">300</span>
                        </div>
                        <div id="t2i-settings">
                            <div class="settings-divider"></div>
                            <div class="slider-row">
                                <label>CFG Scale</label>
                                <input type="range" id="custom-cfg-scale" min="1" max="20" step="0.5" value="9.5">
                                <span class="slider-val" id="custom-cfg-scale-val">9.5</span>
                            </div>
                            <div class="slider-row">
                                <label>Steps</label>
                                <input type="range" id="custom-steps" min="10" max="150" step="5" value="80">
                                <span class="slider-val" id="custom-steps-val">80</span>
                            </div>
                            <div class="slider-row">
                                <label>Alpha</label>
                                <input type="range" id="custom-alpha" min="0" max="1" step="0.05" value="0.5">
                                <span class="slider-val" id="custom-alpha-val">0.5</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="app-statusbar">
            <div class="sb-section" id="sb-status">Ready</div>
            <div class="sb-section sb-mode" id="sb-mode-label">Text to Image</div>
            <div class="sb-section sb-fixed" id="sb-seed">Seed: 0</div>
        </div>
    </div>
    """)

    run_btn = gr.Button("Run", elem_id="gradio-run-btn")

    demo.load(fn=None, js=init_js)
    demo.load(fn=None, js=wire_outputs_js)

    run_btn.click(
        fn=run_inference,
        inputs=[
            mode_input, hidden_image_b64, prompt_input, seed, randomize_seed,
            temperature, max_length, cfg_scale, num_inference_steps, alpha,
        ],
        outputs=[result_image, result_text, seed],
        js=r"""(mode, b64, prompt, s, rs, temp, ml, cfg, steps, a) => {
            var currentMode = window.__currentMode || 'Text-to-Image';
            return [currentMode, b64, prompt, s, rs, temp, ml, cfg, steps, a];
        }""",
    )

if __name__ == "__main__":
    demo.launch(css=css, ssr_mode=False, show_error=True)