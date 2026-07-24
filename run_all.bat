@echo off
echo ===================================================
echo 1. Checking / Finishing Installations
call setup.bat

echo.
echo ===================================================
echo 2. Downloading Gojo Audio Clip
call venv\Scripts\activate.bat
python download_gojo.py

echo.
echo ===================================================
echo 3. Running F5-TTS Zero-Shot Clone Test
python app.py --test

echo.
echo ===================================================
echo Test complete! 
echo Check the output file at: e:\project\searching\anime_voice_cloner\output_cloned.wav
echo.
echo To start the web UI, run:
echo python app.py
echo ===================================================
pause
