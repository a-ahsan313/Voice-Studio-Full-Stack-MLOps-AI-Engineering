@echo off
echo =========================================
echo Setting up Local Anime Voice Cloner
echo =========================================

echo Creating virtual environment...
python -3.11 -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing other dependencies...
pip install -r requirements.txt

echo =========================================
echo Setup Complete!
echo =========================================
