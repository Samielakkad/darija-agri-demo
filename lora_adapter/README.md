---
base_model: Qwen/Qwen2.5-VL-3B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
language:
- ary
- ar
tags:
- lora
- qwen2.5-vl
- agriculture
- darija
---

# Darija crop-ID LoRA for Qwen2.5-VL-3B-Instruct

LoRA adapter that tunes Qwen2.5-VL-3B-Instruct to identify agricultural crops from photos and respond in Moroccan Darija (ary).

- **Task:** crop identification (30 classes) + Darija description
- **Eval:** crop top-1 0.443 on the held-out split (alias-matched, Arabic-normalized — see `crop_aliases.json` at the repo root)
- **Config:** r=8, α=16, dropout 0.05, targets q/k/v/o projections + MLP gate/up/down across layers
- **Trained by:** Sami El Akkad, Tsinghua SIGS — thesis work on low-resource agricultural advisory

## Usage

Load the base model, then apply this adapter with peft. `app_demo.py` at the repo root is the reference implementation (4-bit NF4 on GPU, fp32 on CPU, greedy decoding with fixed seed).

## Limitations

- No disease or treatment supervision — the demo hard-gates those outputs rather than letting the model guess.
- Coverage is 30 crop classes; anything else should hit the refusal path.
- Darija output quality varies with image quality and phrasing.
