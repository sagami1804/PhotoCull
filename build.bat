@echo off
rem ===== PhotoCull build script =====
rem Requires Python 3.10+. Creates dist\PhotoCull.exe
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\PhotoCull\venv"
if not exist "%VENV%\Scripts\python.exe" python -m venv "%VENV%"
if errorlevel 1 goto err
"%VENV%\Scripts\python.exe" -m pip install -q --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install -q -r requirements.txt
if errorlevel 1 goto err
"%VENV%\Scripts\python.exe" -c "import cv2;print(cv2.data.haarcascades)" > "%TEMP%\photocull_casc.txt"
set /p CASC=<"%TEMP%\photocull_casc.txt"
"%VENV%\Scripts\pyinstaller.exe" --noconfirm --clean --onefile --windowed ^
  --name PhotoCull --icon "%~dp0assets\icon.ico" ^
  --workpath "%TEMP%\PhotoCull-build" --specpath "%TEMP%\PhotoCull-build" ^
  --distpath "%~dp0dist" ^
  --add-data "%CASC%haarcascade_eye_tree_eyeglasses.xml;cascades" ^
  --add-data "%~dp0photocull\models\yolo11s-seg.onnx;models" ^
  --add-data "%~dp0photocull\models\face_detection_yunet_2023mar.onnx;models" ^
  --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtQml ^
  --exclude-module PySide6.QtQuick --exclude-module PySide6.Qt3DCore ^
  --exclude-module PySide6.QtMultimedia --exclude-module PySide6.QtPdf ^
  --exclude-module tkinter ^
  "%~dp0main.py"
if errorlevel 1 goto err
echo.
echo Done: %~dp0dist\PhotoCull.exe
exit /b 0
:err
echo Build failed.
exit /b 1
