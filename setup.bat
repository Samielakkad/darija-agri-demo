@echo off
echo === Darija Agricultural VLM - Setup ===
echo Installing Python dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft gradio qwen-vl-utils bitsandbytes Pillow
echo.
echo === Setup complete! Run run.bat to launch the demo ===
pause