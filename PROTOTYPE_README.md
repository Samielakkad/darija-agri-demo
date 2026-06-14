# PROTOTYPE README -- Darija Agricultural VLM Demo
## READ THIS FIRST BEFORE CONTINUING

## STATUS: app_demo.py is DONE (209 lines, SYNTAX_OK). Next = install deps + test + launch.

## KEY PATHS
Base VLM: /mnt/workspace/models/Qwen2.5-VL-3B-Instruct
LoRA:     /mnt/workspace/qwen25_vl_3b_agri_darija_lora/adapter/
Demo:     /mnt/workspace/darija_vlm_demo/app_demo.py (209 lines)
Aliases:  /mnt/workspace/darija_vlm_demo/crop_aliases.json (30 classes)
Images:   /mnt/workspace/Agricultural-crops/ (find *.jpg)
GitHub:   https://github.com/Samielakkad/thesis-code-2

## NEXT STEPS IN ORDER
1. pip show gradio peft bitsandbytes transformers accelerate qwen-vl-utils 2>&1 | grep -E Name
2. pip install -q gradio peft bitsandbytes accelerate qwen-vl-utils --upgrade 2>&1 | tail -5
3. mkdir -p /mnt/workspace/darija_vlm_demo/examples
4. find /mnt/workspace/Agricultural-crops -name *.jpg | shuf | head -5  # pick 2-3 crop images
5. cp chosen images to examples/ folder
6. cd /mnt/workspace/darija_vlm_demo && python3 app_demo.py 2>&1 | tee launch.log
7. echo cd /mnt/workspace/darija_vlm_demo && python3 app_demo.py > run.sh && chmod +x run.sh
8. Push darija_vlm_demo/ to thesis-code-2 GitHub repo

## RULES - NEVER BREAK
- NEVER fabricate disease/treatment. Disease card = greyed Phase2 always.
- Refusal gate: no crop match -> show: mansh mta2kd sol khabir flahi
- Model: 4-bit NF4, do_sample=False, seed=42
- One file: app_demo.py only
- No weights in git (.gitignore has *.safetensors *.bin *.pt)

## APP_DEMO.PY STRUCTURE (209 lines)
normalize_arabic() + crop_match() (exact eval replica)
load_model() = Qwen2.5-VL-3B + LoRA in 4bit NF4
predict() = inference + refusal gate
build_ui() = Gradio Blocks, title=Darija Agricultural Advisor Proposal Demo Phase1
main = load_model() + launch(share=True, server_name=0.0.0.0)

## USER PREFS
Impatient. Just execute. No long explanations. Report: load time + share URL + test results.