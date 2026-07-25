FROM python:3.11-slim

# ffmpeg is required by pydub for every audio operation in the app.
# git is required because some pip packages (f5-tts, rvc-python) pull
# dependencies straight from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so Docker caches this layer and doesn't
# re-download ~1GB+ of torch on every rebuild when you only change app.py
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# saved_voices / rvc_models / training_data get volume-mounted in
# docker-compose.yml tomorrow so data survives container restarts -
# these mkdirs just make sure the paths exist even if run without compose
RUN mkdir -p saved_voices rvc_models training_data temp hf_cache

EXPOSE 7860

# Gradio serves its UI at "/" - a 200 there means the app is genuinely up,
# not just that the process hasn't crashed yet. start-period gives it room
# for slow first-load model downloads on CPU.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app.py"]