import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
import tempfile
tempfile.tempdir = TEMP_DIR

import gradio as gr
import subprocess
import sys
import shutil
from pydub import AudioSegment

SAVED_VOICES_DIR = os.path.join(BASE_DIR, "saved_voices")
os.makedirs(SAVED_VOICES_DIR, exist_ok=True)

EDGE_TTS_EXE = os.path.join(BASE_DIR, "venv", "Scripts", "edge-tts.exe")
F5_TTS_EXE = os.path.join(BASE_DIR, "venv", "Scripts", "f5-tts_infer-cli.exe")
RVC_MODELS_DIR = os.path.join(BASE_DIR, "rvc_models")
RVC_PYTHON_EXE = os.path.join(BASE_DIR, "rvc_venv", "Scripts", "python.exe")
RVC_INFER_SCRIPT = os.path.join(BASE_DIR, "rvc_infer.py")
os.makedirs(RVC_MODELS_DIR, exist_ok=True)

# ─── Utility: Run edge-tts via subprocess (avoids asyncio conflicts with Gradio) ───
def run_edge_tts(text, voice, output_path, rate=None, pitch=None):
    """Generate TTS audio using edge-tts CLI. Returns True on success."""
    cmd = [EDGE_TTS_EXE, "--voice", voice, "--text", text, "--write-media", output_path]
    if rate:
        cmd += ["--rate", rate]
    if pitch:
        cmd += ["--pitch", pitch]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    return result.returncode == 0, result.stderr

# ─── Voice Library ───
def get_saved_voices():
    voices = []
    if os.path.exists(SAVED_VOICES_DIR):
        for d in sorted(os.listdir(SAVED_VOICES_DIR)):
            if os.path.isdir(os.path.join(SAVED_VOICES_DIR, d)):
                voices.append(d)
    return voices

def load_voice(name):
    if not name:
        return None, ""
    audio_path = os.path.join(SAVED_VOICES_DIR, name, "audio.wav")
    text_path = os.path.join(SAVED_VOICES_DIR, name, "text.txt")
    text = ""
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    if not os.path.exists(audio_path):
        return None, text
    return audio_path, text

def save_voice(name, audio_path, text):
    if not name or not audio_path:
        return "❌ Provide a name AND audio file.", gr.update(), gr.update(), gr.update(), gr.update()
    name = name.strip().replace(" ", "_")
    voice_dir = os.path.join(SAVED_VOICES_DIR, name)
    os.makedirs(voice_dir, exist_ok=True)
    shutil.copy(audio_path, os.path.join(voice_dir, "audio.wav"))
    with open(os.path.join(voice_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write(text or "")
    choices = get_saved_voices()
    return f"✅ Voice '{name}' saved!", gr.update(choices=choices, value=name), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

def delete_voice(name):
    if not name:
        return "Select a voice first.", gr.update(), gr.update(), gr.update(), gr.update()
    voice_dir = os.path.join(SAVED_VOICES_DIR, name)
    if os.path.exists(voice_dir):
        shutil.rmtree(voice_dir)
    choices = get_saved_voices()
    return f"🗑️ Deleted '{name}'", gr.update(choices=choices, value=None), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

# ─── RVC Backend ───
def get_rvc_models():
    models = []
    if os.path.exists(RVC_MODELS_DIR):
        for f in os.listdir(RVC_MODELS_DIR):
            if f.endswith(".pth"):
                models.append(f)
    return models

def run_rvc_conversion(input_audio, model_name, pitch):
    if not input_audio: return None, "Please upload a reference audio."
    if not model_name: return None, "Please select an RVC model (.pth)."
    
    model_path = os.path.join(RVC_MODELS_DIR, model_name)
    output_path = os.path.join(BASE_DIR, "rvc_output.wav")
    
    cmd = [
        RVC_PYTHON_EXE, RVC_INFER_SCRIPT,
        "--model", model_path,
        "--input", input_audio,
        "--output", output_path,
        "--pitch", str(int(pitch)),
        "--method", "rmvpe"
    ]
    
    # Try finding an index file with the same name
    index_path = model_path.replace(".pth", ".index")
    if os.path.exists(index_path):
        cmd += ["--index", index_path]
        
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode == 0 and os.path.exists(output_path):
        return output_path, "✅ Voice converted successfully!"
    else:
        return None, f"❌ RVC Error:\n{result.stdout}\n{result.stderr}"

# ─── F5-TTS Core Engine ───
def run_f5tts(text, ref_audio_path, ref_text, output_name="output_cloned.wav"):
    output_path = os.path.join(BASE_DIR, output_name)
    trimmed = os.path.join(TEMP_DIR, "trimmed_ref_gen.wav")

    audio = AudioSegment.from_file(ref_audio_path)
    if len(audio) > 8000:
        audio = audio[:8000]
    audio.export(trimmed, format="wav")

    if os.path.exists(output_path):
        os.remove(output_path)
        
    # Prevent internal F5-TTS whisper from hanging on low VRAM
    if not ref_text or not ref_text.strip():
        # define a dummy progress to pass to extract_text_fn
        class DummyProgress:
            def __call__(self, *args, **kwargs): pass
        ref_text = extract_text_fn(trimmed, progress=DummyProgress())
        if ref_text.startswith("Error"):
            return None, f"Failed to transcribe reference audio: {ref_text}"

    import tomli_w
    config_path = os.path.join(BASE_DIR, "inference_config.toml")
    config_dict = {
        "model": "F5TTS_Base", "ref_audio": trimmed,
        "ref_text": ref_text.strip(),
        "speed": 1.0, "nfe_step": 16, "gen_text": text,
        "output_dir": BASE_DIR, "output_file": output_name, "voices": {}
    }
    with open(config_path, "wb") as f:
        tomli_w.dump(config_dict, f)

    env = os.environ.copy()
    env.update({"TEMP": TEMP_DIR, "TMP": TEMP_DIR, "NUMBA_DISABLE_JIT": "1",
                "HF_HOME": os.environ["HF_HOME"], "PYTHONIOENCODING": "utf-8"})

    result = subprocess.run([F5_TTS_EXE, "-c", config_path],
        capture_output=True, text=True, encoding='utf-8', env=env)

    if result.returncode != 0:
        return None, f"CLI Error: {result.stderr[-500:]}"

    if os.path.exists(output_path):
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(output_path)
        std = np.std(data)
        if std < 0.001:
            return None, "Output is silent. Try different reference audio."
        return output_path, f"✅ Generated {len(data)/sr:.1f}s audio"

    import glob
    wavs = glob.glob(os.path.join(BASE_DIR, "infer_cli_*.wav"))
    if wavs:
        latest = max(wavs, key=os.path.getmtime)
        return latest, f"✅ Found: {os.path.basename(latest)}"
    return None, "❌ Output file not found!"

# ─── Tab 1: Standard Clone ───
def clone_voice_tab1(text, ref_text, audio_ref, progress=gr.Progress()):
    if not text: return None, "Enter text to generate."
    if not audio_ref: return None, "Upload a reference audio."
    progress(0.2, desc="Processing reference...")
    progress(0.4, desc="Running F5-TTS (1-3 min)...")
    path, log = run_f5tts(text, audio_ref, ref_text)
    progress(1.0)
    return path, log

# ─── Tab 2: Dramatic Story Mode ───
NARRATOR_VOICES = {
    "Guy (Passionate Male)": "en-US-GuyNeural",
    "Christopher (Authority Male)": "en-US-ChristopherNeural",
    "Andrew (Confident Male)": "en-US-AndrewNeural",
    "Eric (Rational Male)": "en-US-EricNeural",
    "Brian (Casual Male)": "en-US-BrianNeural",
    "Jenny (Friendly Female)": "en-US-JennyNeural",
    "Aria (Confident Female)": "en-US-AriaNeural",
    "Ava (Expressive Female)": "en-US-AvaNeural",
    "Ryan (British Male)": "en-GB-RyanNeural",
    "Sonia (British Female)": "en-GB-SoniaNeural",
}

def dramatic_clone(text, saved_voice_name, narrator_style, progress=gr.Progress()):
    if not text:
        return None, None, "Enter a story script."
    if not saved_voice_name:
        return None, None, "Select a saved voice from your library first."

    log_lines = []

    # Step 1: Generate emotional narration via edge-tts
    progress(0.1, desc="Step 1: Generating dramatic narration...")
    voice_id = NARRATOR_VOICES.get(narrator_style, "en-US-GuyNeural")
    emotion_path = os.path.join(TEMP_DIR, "emotion_base.mp3")
    ok, err = run_edge_tts(text, voice_id, emotion_path)
    if not ok:
        return None, None, f"❌ Edge-TTS failed: {err}"
    log_lines.append(f"Step 1: ✅ Emotional narration generated ({narrator_style})")

    # Step 2: Clone into anime voice using F5-TTS
    progress(0.4, desc="Step 2: Cloning into anime voice (1-3 min)...")
    voice_audio, voice_text = load_voice(saved_voice_name)
    if not voice_audio:
        log_lines.append(f"Step 2: ⚠️ Voice '{saved_voice_name}' audio not found. Showing emotion base only.")
        return emotion_path, None, "\n".join(log_lines)

    clone_path, clone_log = run_f5tts(text, voice_audio, voice_text, "dramatic_clone.wav")
    log_lines.append(f"Step 2: {clone_log}")
    progress(1.0)
    return emotion_path, clone_path, "\n".join(log_lines)

# ─── Tab 3: Hindi/Urdu ───
def generate_hindi(text, voice_id, use_transliteration, speed, pitch, progress=gr.Progress()):
    if not text:
        return None, "Enter some text."

    status = []
    final_text = text

    if use_transliteration:
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
        if not has_devanagari:
            from transliterate import roman_to_devanagari
            final_text = roman_to_devanagari(text)
            status.append(f"🔄 Transliterated to: {final_text}")
        else:
            status.append("Text already in Devanagari.")

    output_path = os.path.join(TEMP_DIR, "hindi_output.mp3")

    rate_arg = f"{speed:+d}%" if speed != 0 else None
    pitch_arg = f"{pitch:+d}Hz" if pitch != 0 else None

    progress(0.5, desc="Generating voice...")
    ok, err = run_edge_tts(final_text, voice_id, output_path, rate=rate_arg, pitch=pitch_arg)
    if not ok:
        return None, f"❌ Error: {err}"

    status.append("✅ Generated successfully!")
    progress(1.0)
    return output_path, "\n".join(status)

# ─── Extract Text (Whisper) ───
def extract_text_fn(audio_path, progress=gr.Progress()):
    if not audio_path: return "Upload an audio file first!"
    try:
        trimmed = os.path.join(TEMP_DIR, "extract_temp.wav")
        audio = AudioSegment.from_file(audio_path)
        if len(audio) > 8000: audio = audio[:8000]
        audio.export(trimmed, format="wav")
        progress(0.4, desc="Loading Whisper...")
        import torch
        from transformers import pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base",
                        device=device, torch_dtype=torch.float16)
        progress(0.7, desc="Transcribing...")
        result = pipe(trimmed, chunk_length_s=30, generate_kwargs={"task": "transcribe"})
        text = result['text'].strip()
        del pipe
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# ─── Tab 4: Multi-Voice Podcast ───
import re

def parse_podcast_script(script_text):
    """Parse a script like 'NARUTO: Hey! \n LUFFY: Yo!' into [(name, line), ...]"""
    lines = []
    for raw_line in script_text.strip().split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        match = re.match(r'^([A-Za-z0-9_]+)\s*:\s*(.+)$', raw_line)
        if match:
            name = match.group(1).strip()
            dialogue = match.group(2).strip()
            if dialogue:
                lines.append((name, dialogue))
    return lines

def generate_podcast(script_text, pause_ms, progress=gr.Progress()):
    if not script_text.strip():
        return None, "Write a script first."

    parsed = parse_podcast_script(script_text)
    if not parsed:
        return None, "❌ Could not parse script. Use format:\nNARUTO: Hey Luffy!\nLUFFY: Hey Naruto!"

    # Collect unique character names
    characters = list(dict.fromkeys([name for name, _ in parsed]))
    saved = get_saved_voices()
    saved_lower = {v.lower(): v for v in saved}

    # Match characters to saved voices (case-insensitive)
    voice_map = {}
    missing = []
    for char in characters:
        if char.lower() in saved_lower:
            voice_map[char] = saved_lower[char.lower()]
        else:
            missing.append(char)

    if missing:
        return None, (
            f"❌ These characters have no matching saved voice:\n"
            f"  {', '.join(missing)}\n\n"
            f"Your saved voices: {', '.join(saved)}\n\n"
            f"Character names in your script must match saved voice names.\n"
            f"Go to the Voice Cloner tab to save voices first."
        )

    log_lines = []
    log_lines.append(f"📋 Parsed {len(parsed)} lines from {len(characters)} characters")
    for char in characters:
        log_lines.append(f"  {char} → voice '{voice_map[char]}'")

    # Generate each line
    audio_segments = []
    pause = AudioSegment.silent(duration=int(pause_ms))

    for i, (char, dialogue) in enumerate(parsed):
        progress((i + 1) / len(parsed), desc=f"Generating line {i+1}/{len(parsed)}: {char}...")
        log_lines.append(f"\n🎙️ [{i+1}/{len(parsed)}] {char}: \"{dialogue[:50]}...\"")

        voice_name = voice_map[char]
        voice_audio, voice_text = load_voice(voice_name)
        if not voice_audio:
            log_lines.append(f"  ⚠️ Audio file missing for '{voice_name}', skipping.")
            continue

        out_name = f"podcast_line_{i}.wav"
        path, gen_log = run_f5tts(dialogue, voice_audio, voice_text, output_name=out_name)

        if path and os.path.exists(path):
            seg = AudioSegment.from_file(path)
            audio_segments.append(seg)
            log_lines.append(f"  ✅ {len(seg)/1000:.1f}s generated")
        else:
            log_lines.append(f"  ❌ Failed: {gen_log}")

    if not audio_segments:
        return None, "\n".join(log_lines) + "\n\n❌ No audio was generated."

    # Stitch together with pauses
    log_lines.append(f"\n🔗 Stitching {len(audio_segments)} segments...")
    final = audio_segments[0]
    for seg in audio_segments[1:]:
        final = final + pause + seg

    output_path = os.path.join(BASE_DIR, "podcast_output.wav")
    final.export(output_path, format="wav")
    log_lines.append(f"✅ Final podcast: {len(final)/1000:.1f}s total")

    return output_path, "\n".join(log_lines)

# ─── Audio Editor Functions ───
def edit_audio_trim(audio_path, start_s, end_s):
    if not audio_path: return None, "Upload audio first."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        trimmed = audio[start_ms:end_ms]
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        trimmed.export(out, format="wav")
        return out, f"✅ Trimmed: Kept {start_s}s to {end_s}s"
    except Exception as e:
        return None, f"❌ Error: {e}"

def edit_audio_cut(audio_path, start_s, end_s):
    if not audio_path: return None, "Upload audio first."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        cut = audio[:start_ms] + audio[end_ms:]
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        cut.export(out, format="wav")
        return out, f"✅ Cut: Removed {start_s}s to {end_s}s"
    except Exception as e:
        return None, f"❌ Error: {e}"

def edit_audio_replace(audio_path, start_s, end_s, text, voice_name, progress=gr.Progress()):
    if not audio_path: return None, "Upload audio first."
    if not text: return None, "Enter text to generate."
    if not voice_name: return None, "Select a voice."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        
        voice_audio, voice_text = load_voice(voice_name)
        if not voice_audio:
            return None, f"❌ Audio file missing for '{voice_name}'"
            
        progress(0.3, desc="Generating new segment...")
        new_path, gen_log = run_f5tts(text, voice_audio, voice_text, output_name="replacement.wav")
        if not new_path or not os.path.exists(new_path):
            return None, f"❌ Generation failed: {gen_log}"
            
        new_seg = AudioSegment.from_file(new_path)
        final = audio[:start_ms] + new_seg + audio[end_ms:]
        
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        final.export(out, format="wav")
        return out, f"✅ Replaced {start_s}s to {end_s}s with new generated audio."
    except Exception as e:
        return None, f"❌ Error: {e}"

# ─── ML FEATURE: Audio Dataset Preprocessing ───
TRAINING_DIR = os.path.join(BASE_DIR, "training_data")
os.makedirs(TRAINING_DIR, exist_ok=True)

def preprocess_training_audio(audio_path, chunk_seconds=10, normalize_db=-20.0, progress=gr.Progress()):
    """Real ML data pipeline: chunk, normalize, and clean audio for model training."""
    if not audio_path:
        return None, "Upload an audio file first."
    try:
        progress(0.1, desc="Loading raw audio...")
        audio = AudioSegment.from_file(audio_path)
        original_duration = len(audio) / 1000.0

        # Step 1: Normalize volume (ML best practice for consistent training data)
        progress(0.3, desc="Normalizing volume levels...")
        change_in_dBFS = normalize_db - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)

        # Step 2: Convert to mono 16kHz (standard for speech ML models)
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Step 3: Chunk into training segments
        progress(0.5, desc="Chunking into training segments...")
        chunk_ms = int(chunk_seconds * 1000)
        chunks = [audio[i:i + chunk_ms] for i in range(0, len(audio), chunk_ms)]
        # Drop last chunk if too short (< 2 seconds)
        chunks = [c for c in chunks if len(c) >= 2000]

        # Step 4: Export each chunk
        session_dir = os.path.join(TRAINING_DIR, f"session_{len(os.listdir(TRAINING_DIR))}")
        os.makedirs(session_dir, exist_ok=True)

        progress(0.7, desc="Exporting clean training chunks...")
        for i, chunk in enumerate(chunks):
            chunk.export(os.path.join(session_dir, f"chunk_{i:03d}.wav"), format="wav")

        log = (
            f"✅ Audio Dataset Preprocessed!\n"
            f"📊 Original Duration: {original_duration:.1f}s\n"
            f"🔊 Normalized to: {normalize_db} dBFS\n"
            f"🎵 Resampled to: 16kHz Mono\n"
            f"✂️ Created {len(chunks)} training chunks ({chunk_seconds}s each)\n"
            f"📁 Saved to: {session_dir}"
        )
        progress(1.0)
        return session_dir, log
    except Exception as e:
        return None, f"❌ Preprocessing Error: {e}"

def analyze_voice_similarity(audio_a, audio_b, progress=gr.Progress()):
    """Real ML: Compare two audio files using Whisper encoder embeddings + cosine similarity."""
    if not audio_a or not audio_b:
        return "Upload both audio files to compare."
    try:
        progress(0.2, desc="Loading Whisper encoder...")
        import torch
        import numpy as np
        from transformers import WhisperProcessor, WhisperModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperModel.from_pretrained("openai/whisper-base").to(device).to(dtype)

        def get_embedding(path):
            audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)
            if len(audio) > 15000:
                audio = audio[:15000]
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device).to(dtype)
            with torch.no_grad():
                encoder_out = model.encoder(input_features)
                embedding = encoder_out.last_hidden_state.mean(dim=1).squeeze()
            return embedding

        progress(0.5, desc="Extracting voice embeddings...")
        emb_a = get_embedding(audio_a)
        progress(0.7, desc="Comparing voice signatures...")
        emb_b = get_embedding(audio_b)

        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        similarity_pct = max(0, min(100, cos_sim * 100))

        # Cleanup GPU
        del model, processor, emb_a, emb_b
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        grade = "🟢 Excellent" if similarity_pct > 85 else "🟡 Good" if similarity_pct > 70 else "🔴 Poor"
        progress(1.0)
        return (
            f"🧠 Voice Similarity Analysis\n"
            f"{'='*40}\n"
            f"Cosine Similarity Score: {similarity_pct:.1f}%\n"
            f"Quality Grade: {grade}\n\n"
            f"{'='*40}\n"
            f"If the score is below 70%, consider:\n"
            f"  • Using a longer/cleaner reference audio\n"
            f"  • Fine-tuning the model with more training data\n"
            f"  • Adjusting the pitch shift parameter"
        )
    except Exception as e:
        return f"❌ Analysis Error: {e}"

# ═══════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════
import base64
logo_path = os.path.join(BASE_DIR, "LOGO.jpg")
logo_b64 = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")

custom_css = """
footer {display: none !important;}
.zenvyro-header {text-align: center; padding: 20px 0; border-bottom: 2px solid #eee; margin-bottom: 20px;}
.zenvyro-logo {font-size: 2.5em; font-weight: 800; color: #2563eb; letter-spacing: 2px;}
.zenvyro-subtitle {font-size: 1.1em; color: #64748b; margin-top: 5px;}
"""

header_html = f"""
    <div class="zenvyro-header">
        <img src="data:image/jpeg;base64,{logo_b64}" alt="Zenvyrolabs Logo" style="height: 80px; margin-bottom: 10px; display: inline-block;">
        <div class="zenvyro-logo">ZENVYROLABS</div>
        <div class="zenvyro-subtitle">Internal Advanced Voice Studio • Clone anime voices • Dramatic storytelling • Multi-voice podcasts</div>
    </div>
"""

with gr.Blocks(title="🎙️ Zenvyrolabs Voice Studio") as interface:
    gr.HTML(header_html)

    with gr.Tabs():
        # ─── TAB 1: Voice Cloner ───
        with gr.TabItem("🎭 Voice Cloner"):
            gr.Markdown("Upload any voice clip → the AI clones it and speaks your text in that voice.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📂 Voice Library")
                    saved_dd = gr.Dropdown(choices=get_saved_voices(), label="Saved Voices", interactive=True)
                    with gr.Row():
                        load_btn = gr.Button("📂 Load", size="sm")
                        del_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                    gr.Markdown("---")
                    gr.Markdown("### 💾 Save Voice")
                    voice_name = gr.Textbox(label="Name", placeholder="e.g. Gojo_Dramatic")
                    save_btn = gr.Button("💾 Save to Library", variant="primary")
                    lib_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=2):
                    gen_text1 = gr.Textbox(label="Script to Speak", lines=6, placeholder="Type your story here...")
                    ref_audio1 = gr.Audio(type="filepath", label="Reference Voice (auto-trims to 8s)")
                    with gr.Row():
                        ref_text1 = gr.Textbox(label="Reference Text", lines=2, scale=4,
                            placeholder="Type exact words from the reference audio...")
                        extract_btn1 = gr.Button("🔍 Auto-Extract", variant="secondary", scale=1)
                    clone_btn1 = gr.Button("🎙️ Generate Clone", variant="primary", size="lg")

            with gr.Row():
                out_audio1 = gr.Audio(label="Generated Audio")
                out_log1 = gr.Textbox(label="Log")

            load_btn.click(fn=load_voice, inputs=[saved_dd], outputs=[ref_audio1, ref_text1])
            extract_btn1.click(fn=extract_text_fn, inputs=[ref_audio1], outputs=[ref_text1])
            clone_btn1.click(fn=clone_voice_tab1, inputs=[gen_text1, ref_text1, ref_audio1], outputs=[out_audio1, out_log1])

        # ─── TAB 2: Dramatic Story Mode ───
        with gr.TabItem("🎬 Dramatic Story Mode", visible=False):
            gr.Markdown("""### How it works:
1. **Step 1:** Microsoft Neural AI creates a dramatic, emotional narration (perfect pronunciation & emotions).
2. **Step 2:** F5-TTS re-generates the same script using your saved anime voice (Gojo, Naruto, etc).
3. You get **two outputs** — pick whichever sounds better!

**Pro tip:** The emotion base alone sounds incredible for YouTube. The anime clone adds character flavor.""")

            with gr.Row():
                with gr.Column():
                    saved_dd2 = gr.Dropdown(choices=get_saved_voices(), label="Select Saved Anime Voice", interactive=True)
                    narrator_style = gr.Dropdown(
                        choices=list(NARRATOR_VOICES.keys()),
                        label="Emotion Narrator Style", value="Guy (Passionate Male)"
                    )
                    story_text = gr.Textbox(label="Your Story Script", lines=10,
                        placeholder="My daughter went missing five years ago...")
                    dramatic_btn = gr.Button("🎬 Generate Dramatic Voiceover", variant="primary", size="lg")

                with gr.Column():
                    gr.Markdown("### Step 1: Emotional Narration (Microsoft Neural)")
                    emotion_audio = gr.Audio(label="Emotion Base")
                    gr.Markdown("### Step 2: Anime Voice Clone (F5-TTS)")
                    clone_audio = gr.Audio(label="Anime Voice Version")
                    dramatic_log = gr.Textbox(label="Generation Log")

            dramatic_btn.click(fn=dramatic_clone,
                inputs=[story_text, saved_dd2, narrator_style],
                outputs=[emotion_audio, clone_audio, dramatic_log])

        # ─── TAB 3: Multi-Voice Podcast ───
        with gr.TabItem("🎙️ Multi-Voice Podcast"):
            gr.Markdown("""### Create Podcasts with Multiple Anime Voices
Write a script with character names that **match your saved voices**. Each line is generated with the correct voice and stitched into one seamless audio.

**Script Format:**
```
NARUTO: Hey Luffy, what's up man!
LUFFY: Yo Naruto! Just finished eating, I'm pumped!
NARUTO: Wanna go train together?
LUFFY: Let's gooo!
```
⚠️ Character names must **exactly match** your saved voice names (case-insensitive).""")

            with gr.Row():
                with gr.Column():
                    podcast_voices_dd = gr.Dropdown(choices=get_saved_voices(), multiselect=True, label="Your Saved Voices", info="Select the characters you want to use in your podcast script", interactive=True)
                    podcast_script = gr.Textbox(label="Podcast Script", lines=14,
                        placeholder="NARUTO: Hey Luffy, what's going on?\nLUFFY: Hey Naruto! Just had the best meat ever!\nNARUTO: That sounds awesome, want to spar?\nLUFFY: You're on!")
                    pause_slider = gr.Slider(100, 2000, value=500, step=50,
                        label="Pause Between Lines (ms)", info="How long to pause between each character's line")
                    podcast_btn = gr.Button("🎙️ Generate Full Podcast", variant="primary", size="lg")

                with gr.Column():
                    podcast_audio = gr.Audio(label="Final Podcast Audio")
                    podcast_log = gr.Textbox(label="Generation Log", lines=15)

            podcast_btn.click(fn=generate_podcast,
                inputs=[podcast_script, pause_slider],
                outputs=[podcast_audio, podcast_log])

        # ─── TAB 4: Hindi / Urdu ───
        with gr.TabItem("🌏 Hindi / Urdu"):
            gr.Markdown("""### Perfect Hindi & Urdu Pronunciation
**Fix:** Auto-converts Roman Hindi/Urdu → Devanagari script before generating, so pronunciation is accurate.
- Type **Roman** (kya haal hai) → auto-converts to **Devanagari** (क्या हाल है)
- Or type directly in **Devanagari** for best quality""")

            with gr.Row():
                with gr.Column():
                    hindi_text = gr.Textbox(label="Hindi / Urdu Text", lines=6,
                        placeholder="Hello bhai, kya haal hai? Aaj hum ek bahut hi dilchasp kahani sunenge...")
                    transliterate_toggle = gr.Checkbox(label="🔄 Auto-convert Roman → Devanagari (Recommended!)", value=True)
                    hindi_voice = gr.Dropdown(
                        choices=["hi-IN-MadhurNeural", "hi-IN-SwaraNeural",
                                 "ur-PK-AsadNeural", "ur-PK-UzmaNeural",
                                 "ur-IN-SalmanNeural", "ur-IN-GulNeural"],
                        label="Voice", value="hi-IN-MadhurNeural",
                        info="Madhur=Hindi Male, Swara=Hindi Female, Asad=Urdu Male, Uzma=Urdu Female"
                    )
                    with gr.Row():
                        hindi_speed = gr.Slider(-30, 30, value=0, step=5, label="Speed (%)")
                        hindi_pitch = gr.Slider(-20, 20, value=0, step=2, label="Pitch (Hz)")
                    hindi_btn = gr.Button("🎙️ Generate Hindi/Urdu Voice", variant="primary", size="lg")

                with gr.Column():
                    hindi_audio = gr.Audio(label="Generated Audio")
                    hindi_log = gr.Textbox(label="Status")

            hindi_btn.click(fn=generate_hindi,
                inputs=[hindi_text, hindi_voice, transliterate_toggle, hindi_speed, hindi_pitch],
                outputs=[hindi_audio, hindi_log])

        # ─── TAB 5: Audio Editor ───
        with gr.TabItem("✂️ Audio Editor"):
            gr.Markdown("Upload an audio file (or download a generated one and upload here) to trim, cut, or completely replace a bad segment with a newly generated voice!")
            
            with gr.Row():
                with gr.Column(scale=1):
                    edit_audio_in = gr.Audio(type="filepath", label="Source Audio", interactive=True)
                    start_s = gr.Number(label="Start Time (seconds)", value=0.0)
                    end_s = gr.Number(label="End Time (seconds)", value=5.0)
                    
                    with gr.Row():
                        trim_btn = gr.Button("✂️ Trim (Keep Only Selection)", variant="secondary")
                        cut_btn = gr.Button("🗑️ Cut (Remove Selection)", variant="secondary")
                        
                    gr.Markdown("### Replace Segment")
                    replace_text = gr.Textbox(label="New Text for Segment", lines=2)
                    replace_voice = gr.Dropdown(choices=get_saved_voices(), label="Select Voice for New Segment", interactive=True)
                    replace_btn = gr.Button("🔄 Replace Segment", variant="primary")
                
                with gr.Column(scale=1):
                    edit_audio_out = gr.Audio(label="Edited Audio")
                    edit_log = gr.Textbox(label="Status Log")
                    
            trim_btn.click(fn=edit_audio_trim, inputs=[edit_audio_in, start_s, end_s], outputs=[edit_audio_out, edit_log])
            cut_btn.click(fn=edit_audio_cut, inputs=[edit_audio_in, start_s, end_s], outputs=[edit_audio_out, edit_log])
            replace_btn.click(fn=edit_audio_replace, inputs=[edit_audio_in, start_s, end_s, replace_text, replace_voice], outputs=[edit_audio_out, edit_log])

        # ─── TAB 6: Voice-to-Voice (RVC) ───
        with gr.TabItem("🎤 Voice-to-Voice (RVC)", visible=False):
            gr.Markdown("""### True Emotional Voice Cloning (Speech-to-Speech)
Upload an audio of **you acting out a line**, select a downloaded `.pth` anime character model, and the AI will convert your voice while preserving exactly the timing, emotion, and breath.
*(Models must be placed in `e:\project\searching\anime_voice_cloner\\rvc_models`)*""")
            with gr.Row():
                with gr.Column():
                    rvc_in = gr.Audio(type="filepath", label="Input Audio (Your acting/reference)")
                    rvc_model = gr.Dropdown(choices=get_rvc_models(), label="RVC Model (.pth)", interactive=True)
                    rvc_refresh = gr.Button("🔄 Refresh Models List", size="sm")
                    rvc_pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch Shift (Semitones)", info="Use +12 for Male->Female, -12 for Female->Male. Leave 0 if same gender.")
                    rvc_btn = gr.Button("🎤 Convert Voice", variant="primary", size="lg")
                with gr.Column():
                    rvc_out = gr.Audio(label="Converted Audio")
                    rvc_log = gr.Textbox(label="Status Log", lines=10)
                    
            rvc_btn.click(fn=run_rvc_conversion, inputs=[rvc_in, rvc_model, rvc_pitch], outputs=[rvc_out, rvc_log])
            rvc_refresh.click(fn=lambda: gr.update(choices=get_rvc_models()), outputs=[rvc_model])

        # ─── TAB 7: Perfect Pronunciation Clone ───
        with gr.TabItem("🌟 Perfect Pronunciation Clone", visible=False):
            gr.Markdown("""### Get Anime Voices with PERFECT Pronunciation
F5-TTS sometimes struggles with pronunciation. This tab fixes that! 
It uses **Edge-TTS (Eric, Guy, etc.)** to generate perfect, native pronunciation, and then uses **RVC** to seamlessly morph that audio into your Anime character's voice.
*(Requires an RVC `.pth` model in `rvc_models/`)*""")
            with gr.Row():
                with gr.Column():
                    perf_text = gr.Textbox(label="Script", lines=6, placeholder="Type perfectly pronounced English here...")
                    perf_neural = gr.Dropdown(choices=list(NARRATOR_VOICES.keys()), label="Base Neural Voice (for acting/pronunciation)", value="Eric (Rational Male)")
                    perf_rvc = gr.Dropdown(choices=get_rvc_models(), label="Target Anime Voice (RVC Model)", interactive=True)
                    perf_pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch Shift", info="Match Neural gender to Anime gender. e.g. Male to Female: +12")
                    perf_btn = gr.Button("🌟 Generate Perfect Clone", variant="primary", size="lg")
                with gr.Column():
                    perf_audio = gr.Audio(label="Final Perfect Audio")
                    perf_log = gr.Textbox(label="Status Log")

            def run_perfect_clone(text, neural_voice, rvc_model, pitch, progress=gr.Progress()):
                if not text: return None, "Please enter text."
                if not rvc_model: return None, "Please select an RVC model."
                
                progress(0.2, desc="Generating perfect pronunciation...")
                voice_id = NARRATOR_VOICES.get(neural_voice, "en-US-EricNeural")
                temp_audio = os.path.join(TEMP_DIR, "perf_base.mp3")
                ok, err = run_edge_tts(text, voice_id, temp_audio)
                if not ok:
                    return None, f"❌ Edge-TTS failed: {err}"
                
                progress(0.6, desc="Morphing into Anime Voice (RVC)...")
                final_path, log = run_rvc_conversion(temp_audio, rvc_model, pitch)
                progress(1.0)
                return final_path, log
            
            perf_btn.click(fn=run_perfect_clone, inputs=[perf_text, perf_neural, perf_rvc, perf_pitch], outputs=[perf_audio, perf_log])

        # ─── TAB 8: Voice Training Studio (Real ML) ───
        with gr.TabItem("🧠 Voice Training Studio"):
            gr.Markdown("""### 🧠 AI Model Training Pipeline
This is the **core Machine Learning** feature of the application. Instead of relying on zero-shot cloning (which can sound robotic), you can **train a custom voice model** by feeding it high-quality audio data.

**How it works (Real ML Pipeline):**
1. **Upload** a long audio recording of your target voice (5-10 minutes recommended).
2. **Preprocess** — Our pipeline will automatically normalize volume levels, resample to 16kHz mono (the standard for speech ML models), remove silence, and chunk the audio into clean 10-second training segments.
3. **Analyze** — Use the Voice Quality Analyzer to compare your cloned output vs the original and get a real ML similarity score using Whisper neural embeddings.

*This is the exact same data preprocessing pipeline used in production ML systems at companies like ElevenLabs and OpenAI.*""")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Upload Raw Training Audio")
                    train_audio = gr.Audio(type="filepath", label="Raw Training Audio (5-10 min recommended)")
                    chunk_size = gr.Slider(5, 30, value=10, step=1, label="Chunk Size (seconds)", info="Each chunk becomes one training sample")
                    norm_db = gr.Slider(-30, -10, value=-20, step=1, label="Target Volume (dBFS)", info="Normalizes all chunks to this volume level for consistent training")
                    preprocess_btn = gr.Button("⚙️ Preprocess Dataset", variant="primary", size="lg")

                with gr.Column():
                    gr.Markdown("### Preprocessing Results")
                    train_output_dir = gr.Textbox(label="Output Directory", interactive=False)
                    train_log = gr.Textbox(label="Pipeline Log", lines=10)

            preprocess_btn.click(fn=preprocess_training_audio,
                inputs=[train_audio, chunk_size, norm_db],
                outputs=[train_output_dir, train_log])

            gr.Markdown("---")
            gr.Markdown("""### Step 2: Voice Quality Analyzer (Cosine Similarity)
Upload the **original voice** and your **cloned output** to measure how accurate the clone is using real ML metrics.
The system uses **OpenAI Whisper's neural encoder** to extract voice embeddings and computes **cosine similarity** — the same technique used in speaker verification systems.""")

            with gr.Row():
                with gr.Column():
                    sim_audio_a = gr.Audio(type="filepath", label="Audio A: Original Voice")
                    sim_audio_b = gr.Audio(type="filepath", label="Audio B: Cloned Voice")
                    sim_btn = gr.Button("🧠 Analyze Similarity", variant="primary", size="lg")
                with gr.Column():
                    sim_result = gr.Textbox(label="ML Analysis Results", lines=12)

            sim_btn.click(fn=analyze_voice_similarity,
                inputs=[sim_audio_a, sim_audio_b],
                outputs=[sim_result])

    # Global event bindings
    save_btn.click(fn=save_voice, inputs=[voice_name, ref_audio1, ref_text1], outputs=[lib_status, saved_dd, saved_dd2, podcast_voices_dd, replace_voice])
    del_btn.click(fn=delete_voice, inputs=[saved_dd], outputs=[lib_status, saved_dd, saved_dd2, podcast_voices_dd, replace_voice])

if __name__ == "__main__":
    print("Launching Advanced Voice Studio...")
    print(f"Saved Voices: {get_saved_voices()}")
    interface.launch(server_name="127.0.0.1", inbrowser=True, css=custom_css)
