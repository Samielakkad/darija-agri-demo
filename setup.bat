@echo off
echo === Darija Agricultural VLM - Setup ===
echo Installing Python dependencies...
REM Versions are pinned for reproducible installs. This demo targets a specific
REM API surface: transformers>=4.49 is the first release exposing
REM Qwen2_5_VLForConditionalGeneration, and the torch cu121 trio caps at 2.5.1.
REM An unpinned "pip install transformers" would eventually pull a release that
REM renames/moves that class and silently break LoRA adapter loading.
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0 accelerate==1.4.0 peft==0.15.2 gradio==5.16.0 qwen-vl-utils==0.0.8 bitsandbytes==0.45.0 Pillow==11.1.0
echo.
echo === Setup complete! Run run.bat to launch the demo ===
pause
