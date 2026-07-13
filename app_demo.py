#!/usr/bin/env python3
"""Darija Agricultural Advisor - Proposal Demo (Phase 1).
Wraps Qwen2.5-VL-3B + LoRA adapter (crop top-1 0.443).
HONESTY: only shows crop ID + Darija text + confidence.
Disease & Treatment are Phase 2 (not trained).
"""
import json, re, time
from pathlib import Path
import torch
import gradio as gr
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

from crop_matching import crop_match
from prompting import build_crop_messages

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"  # auto-downloaded from HuggingFace
ADAPTER_DIR = str(HERE / "lora_adapter")
ALIASES_PATH = HERE / "crop_aliases.json"
SEED = 42
MAX_NEW_TOKENS = 1200
USE_GPU = torch.cuda.is_available()

BANNER = (
    "⚠️ **Prototype** — identifies crop + responds in Darija. "
    "Disease & treatment are Phase 2 (not yet trained)."
)

# Load crop aliases
with open(ALIASES_PATH, encoding='utf-8') as f:
    CROP_ALIASES = json.load(f)

model = None
processor = None

def load_model():
    global model, processor
    dev = "GPU" if USE_GPU else "CPU"
    print("Loading model ("+dev+")...", flush=True)
    processor = AutoProcessor.from_pretrained(
        ADAPTER_DIR, trust_remote_code=True,
        min_pixels=256*28*28, max_pixels=1280*28*28,
    )
    if USE_GPU:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, quantization_config=bnb_cfg, device_map="auto",
            trust_remote_code=True)
    else:
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, device_map="cpu",
            trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    print("Model ready on "+dev+".", flush=True)

def predict(image, question):
    if model is None:
        return "[model not loaded]", "", "error"
    if image is None:
        return "", "", "Please upload an image."
    torch.manual_seed(SEED)
    messages = build_crop_messages(image, question)
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS, pad_token_id=processor.tokenizer.eos_token_id)
    raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    darija_text = raw.strip()
    # Parse JSON if present
    m = re.search(r"\{.*\}", darija_text, re.S)
    crop_guess_text = darija_text
    if m:
        try:
            parsed = json.loads(m.group())
            crop_guess_text = parsed.get("crop", "") + " " + parsed.get("description", "")
        except Exception:
            pass
    matched = crop_match(crop_guess_text, CROP_ALIASES)
    if matched:
        confidence = "High (alias matched)"
        crop_out = matched
    else:
        confidence = "Low (no known crop matched)"
        crop_out = "مانش متأكد — سول خبير فلاحي"
    return crop_out, darija_text, confidence

def find_examples(n=3):
    import glob
    ex_dir = HERE / "examples"
    imgs = sorted(glob.glob(str(ex_dir / "*.jpg")))
    return imgs[:n] if imgs else None

def build_ui():
    ex = find_examples(3)
    with gr.Blocks(title="Darija Agricultural Advisor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Darija Agricultural Advisor — Proposal Demo (Phase 1)")
        gr.Markdown(BANNER)
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Crop image")
                q = gr.Textbox(label="Optional crop-ID question (Darija)",
                    placeholder="شنو هاد لمحصول?")
                btn = gr.Button("Analyze", variant="primary")
                if ex:
                    gr.Examples(examples=[[e] for e in ex], inputs=[img],
                        label="Click to try")
            with gr.Column(scale=1):
                crop_out = gr.Textbox(label="🌾 Crop (identified)", interactive=False)
                darija_out = gr.Textbox(label="🗣️ Darija crop description",
                    interactive=False, lines=6)
                conf_out = gr.Textbox(label="✅ Confidence / gate", interactive=False)
                gr.Textbox(label="🌿 Disease & Treatment",
                    value="Phase 2 — not yet trained. No disease supervision in current model.",
                    interactive=False)
        btn.click(predict, inputs=[img, q], outputs=[crop_out, darija_out, conf_out])
    return demo

if __name__ == "__main__":
    t0 = time.time()
    load_model()
    print("Model loaded in "+str(round(time.time()-t0,1))+"s", flush=True)
    build_ui().launch(share=True, server_name="0.0.0.0")
