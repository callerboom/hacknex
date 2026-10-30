@echo off
title EDRA AI BRAIN - Person 3 Module B (EdgeVoice)
color 0A
cls
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
pip install pyyaml numpy psutil sounddevice pyttsx3 -q 2>nul
python demo.py
pause
