# Darija Agricultural VLM — demo

Gradio demo of a vision-language model that identifies a crop from a photo and answers in Moroccan Darija. Part of my master's work at Tsinghua SIGS on low-resource agricultural advisory.

**Model:** [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) + the LoRA adapter in [`lora_adapter/`](./lora_adapter) (r=8, α=16, attention + MLP projections). The base model downloads from Hugging Face on first run (~6 GB); the adapter ships with the repo.

## What it does

- Takes a crop photo, optionally with a crop-identification question in Darija.
- Generates a Darija description — greedy decoding, fixed seed, so runs are reproducible.
- Matches the generated text against 30 crop classes through an Arabic-normalized alias table ([`crop_aliases.json`](./crop_aliases.json)) — the same matcher used in the eval, not a separate heuristic.
- **Refusal gate:** if nothing in the output matches a known crop, the demo answers «مانش متأكد — سول خبير فلاحي» ("not sure — ask an agricultural expert") instead of guessing.
- **Crop-only prompt:** the base instruction always limits the model to crop identity and directly visible features. Optional user text is quoted as untrusted context and cannot replace that instruction; disease, pest, treatment, dosage, and unrelated requests are explicitly out of scope.

Current adapter scores crop top-1 **0.443** on the eval split. Disease and treatment advice are deliberately absent: nothing in the current training covers them, so the UI pins them as a locked Phase 2 panel rather than letting the model improvise medical-grade advice.

The crop-only instruction is a prompt-level scope control, not a medical-safety
classifier. Do not use this prototype to make crop-health or treatment decisions,
and review model output before showing it in any higher-stakes setting.

## Run it

Windows:

```bat
setup.bat   :: installs torch (cu121) + transformers/peft/gradio/bitsandbytes
run.bat
```

Linux / macOS:

```bash
pip install -r requirements.txt
./run.sh
```

With a CUDA GPU the base model loads in 4-bit NF4; without one it falls back to fp32 on CPU (works, slow).

### Network settings

The demo starts on `http://127.0.0.1:7860` and does not request a public share
link by default. Override a setting only when the deployment needs it:

| Environment variable | Default | Purpose |
|---|---:|---|
| `DARIJA_DEMO_HOST` | `127.0.0.1` | Bind address; use `0.0.0.0` only for deliberate network access |
| `DARIJA_DEMO_PORT` | `7860` | TCP port, from `1` through `65535` |
| `DARIJA_DEMO_SHARE` | `false` | Set to `true` to request a Gradio public share link |

Example for access from another device on a trusted local network (Linux/macOS):

```bash
DARIJA_DEMO_HOST=0.0.0.0 DARIJA_DEMO_PORT=7860 ./run.sh
```

PowerShell:

```powershell
$env:DARIJA_DEMO_HOST = "0.0.0.0"
$env:DARIJA_DEMO_PORT = "7860"
.\run.bat
```

Binding to `0.0.0.0` can expose the app to the local network, and a share link
can expose it publicly. This prototype has no built-in authentication; use either
option only behind appropriate network controls. Invalid settings stop startup
before the model is loaded.

## Layout

| path | what |
|---|---|
| `app_demo.py` | model load, inference, refusal gate, and UI |
| `crop_matching.py` | dependency-free Arabic normalization and crop alias matching |
| `prompting.py` | crop-only system policy and optional-question handling |
| `runtime_config.py` | private-by-default, validated Gradio launch settings |
| `lora_adapter/` | LoRA weights + processor/tokenizer config |
| `crop_aliases.json` | 30 crop classes → Arabic/Darija aliases |
| `examples/` | sample images wired into the UI |

## Tests

The matcher tests use only Python's standard library, so they do not load or
download the model:

```bash
python -m unittest discover -s tests -v
```

## Contact

Sami El Akkad — [sam25@mails.tsinghua.edu.cn](mailto:sam25@mails.tsinghua.edu.cn) · [LinkedIn](https://www.linkedin.com/in/samielakkad)
