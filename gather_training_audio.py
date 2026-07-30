"""
gather_training_audio.py

Downloads audio from a list of source URLs for one character, concatenates
them into a single raw file, and reports total duration so you know when
you've hit the 10+ minute target for Task 4.

Usage:
    python gather_training_audio.py "Character Name" URL1 URL2 URL3 ...

Notes:
- Requires yt-dlp + ffmpeg (both already in requirements.txt / the Docker image).
- Prefer solo-dialogue clips (one character talking, no overlapping voices,
  minimal background music) over general clips - the denoise/silence-trim
  pipeline from yesterday helps, but it can't separate two overlapping
  speakers, and cleaner input always beats relying on cleanup after the fact.
- This is for a personal/portfolio project. Keep downloaded source material
  private and don't redistribute it - you're using it to train a local
  voice model for your own internship deliverable, not publishing the clips.
"""
import os
import sys
import subprocess
from pydub import AudioSegment

def gather(character_name, urls, out_dir="raw_audio"):
    char_dir = os.path.join(out_dir, character_name.replace(" ", "_"))
    os.makedirs(char_dir, exist_ok=True)

    clips = []
    for i, url in enumerate(urls):
        out_path = os.path.join(char_dir, f"source_{i:02d}.wav")
        print(f"[{i+1}/{len(urls)}] Downloading: {url}")
        result = subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "wav", "-o", out_path.replace(".wav", ".%(ext)s"), url],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ❌ Failed: {result.stderr.strip()[-300:]}")
            continue
        if os.path.exists(out_path):
            clips.append(out_path)
            dur = len(AudioSegment.from_file(out_path)) / 1000
            print(f"  ✅ Got it: {dur:.1f}s")
        else:
            print(f"  ⚠️ yt-dlp reported success but file wasn't found at {out_path} - check filename/extension")

    if not clips:
        print("\nNo clips downloaded successfully. Nothing to concatenate.")
        return

    print(f"\nConcatenating {len(clips)} clip(s)...")
    combined = AudioSegment.empty()
    for c in clips:
        combined += AudioSegment.from_file(c)

    combined_path = os.path.join(char_dir, f"{character_name.replace(' ', '_')}_raw_combined.wav")
    combined.export(combined_path, format="wav")

    total_min = len(combined) / 1000 / 60
    status = "✅ Target met" if total_min >= 10 else f"⚠️ Need {10 - total_min:.1f} more minutes"
    print(f"\nTotal combined duration: {total_min:.1f} minutes - {status}")
    print(f"Saved to: {combined_path}")
    print(f"\nNext: feed {combined_path} into the Voice Training Studio tab "
          f"(preprocess_training_audio) to denoise/trim/chunk it.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    character = sys.argv[1]
    source_urls = sys.argv[2:]
    gather(character, source_urls)