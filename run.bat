@echo off
echo === Darija Agricultural VLM Demo ===
echo Starting app... (first run downloads ~6GB base model from HuggingFace)
echo.
cd /d %~dp0
python app_demo.py
pause