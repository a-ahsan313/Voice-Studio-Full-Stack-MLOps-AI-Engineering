# 🎙️ Zenvyrolabs Advanced Voice Studio - Engineering Handbook

Welcome to the Zenvyrolabs AI Engineering Internship! We are incredibly excited to see you tackle this project.

Your mission is to take our existing Voice Studio MVP (Minimum Viable Product) and transform it into a highly optimized, robust, production-ready application. We want this app to be incredibly fast, 100% accurate, and extremely easy for our clients to install.

### 🌟 Portfolio Opportunity
This is a highly advanced, full-stack AI project. You have our full permission to remove the Zenvyrolabs logo, replace it with your own branding, and redesign the UI to make it look as premium as possible. Make it your own masterpiece for your resume!

---

## 🧠 The Big Problem: Why Are We Building This?

Before you write any code, you need to understand the two biggest problems in the AI Voice industry right now, and how your project will solve them.

### Problem 1: The Pronunciation Barrier
Imagine you want a Japanese anime character (like Naruto) to speak Hindi ("Kya haal hai"). If you just feed Hindi text directly into a standard AI, the AI will try to read it with a heavy, broken Japanese accent, completely ruining the Hindi pronunciation. 

The Solution: You will build a "Hybrid System". First, we use Microsoft's Neural Voices to generate the text with a *perfect, native Hindi accent*. Then, we use an AI filter to replace the Microsoft voice with Naruto's voice. The result? Naruto speaking perfect, native Hindi!

### Problem 2: The "Robotic Emotion" Barrier
If you only give an AI a 10-second audio clip to clone a voice (this is called "Zero-Shot" cloning), the AI sounds like a flat, boring robot. If that 10-second clip has background noise, the AI will permanently clone the background noise!

The Solution (AI Training): Instead of a 10-second clip, you will build a system that takes a minimum of 10 full minutes of audio and actually *trains* a custom AI model. Why a minimum of 10 minutes? Because the AI needs a massive amount of varied data to fully memorize a character's breathing patterns, their exact laugh, and their deep emotional range. A model trained on a large dataset goes from sounding like a "robot" to sounding like a real human voice actor!

*(👉 Open the `demo.html` file in your browser right now! It will give you a visual, interactive preview of what the AI Training pipeline looks like.)*

---

## 🎯 Detailed Task Breakdown (Total: 200 Points)

We have broken this project down into 4 main tasks. We will not tell you exactly which lines of code to change—you are an engineer now! We will highlight the problems, and it is up to you to figure out how to deliver the expected output.

---

### 🐳 Task 1: Make It Easy To Install (Docker) (80 Points)
The Problem: Right now, the application relies on a `setup.bat` file. If a user has the wrong version of Python installed, the AI libraries crash. If their internet drops while downloading the 2.4GB PyTorch models, the app breaks. It is a nightmare for clients to install.

The Expected Output:
You must containerize this entire application using Docker. 
- You need to write a `Dockerfile` and a `docker-compose.yml` file.
- Anyone in the world should be able to download your project, run one Docker command, and the app should instantly open in their browser without them needing to install Python or FFmpeg manually.
- Make sure you set up "Volumes" so that if the Docker container turns off, the user doesn't lose all their saved voices!

---

### 🎯 Task 2: Fix the Podcast Text & Perfect Pronunciation (40 Points)
The Problem: The app currently has a Multi-Voice Podcast feature where users can type a script (e.g., `NARUTO: Hey guys!`). However, the code parsing this text is very fragile. If a user types `NARUTO : Hey guys` (with an extra space before the colon), the whole app crashes. Also, as mentioned earlier, foreign languages currently sound terrible because they aren't being routed properly.

The Expected Output:
- Crash-Proof Parsing: Rewrite the podcast script logic. It should intelligently figure out which character is speaking, no matter how the user formats the colons or spaces. If they type a character name that doesn't exist, it should show a polite warning message, not crash the server.
- Perfect Pronunciation Routing: Update the voice generation logic so that if the script contains Hindi/Urdu, it *always* generates a perfectly pronounced Microsoft Neural base audio first, before cloning it into the anime character.
- Smooth Audio: Make sure when characters are talking back and forth in the podcast, the audio cuts are smooth and natural, not sudden robotic silences.

---

### 🧠 Task 3: Build the AI Audio Training Pipeline (40 Points)
The Problem: To solve the "Robotic Emotion" problem, clients need to be able to upload large amounts of audio to train their own custom characters. But raw audio from the internet is usually messy, varied in volume, and full of silent gaps. An AI cannot train on messy audio.

The Expected Output:
You must finish upgrading the Voice Training Studio pipeline. We already built the basic code that chops the audio into 10-second chunks and grades the voice quality, but it crashes on messy audio. Your job is to add the filters:
- The Noise Filter: Write code to automatically scan the uploaded audio and filter out background static or music before it gets chopped.
- The Silence Cutter: Write code to automatically detect and cut out any silent "dead air" so the AI doesn't waste time training on nothing.

---

### 🎥 Task 4: The Final 5-Minute Video Submission (40 Points)
The Problem: You have built an incredible, fully-functioning AI engine. Now you need to prove it works in the real world to get your final grade!

The Expected Output:
1. Gather Data: Go online and download a minimum of 10 minutes of clean audio for at least 2 famous fictional characters (e.g., Darth Vader, SpongeBob, Naruto, Iron Man, Harry Potter).
2. Train Models: Run their raw audio through your new ML Preprocessing Pipeline (Task 3) and train their custom AI models.
3. Write a Script: Write a funny or interesting 5-minute podcast script featuring those 2 characters talking to each other.
4. Record a Video: Run your script through the Multi-Voice Podcast tab. Record a screen-capture video (using OBS or Loom) showing the UI working and playing the final 5-minute audio so we can hear the flawless emotions and pronunciation. Submit this video!

---

## 🚀 Getting Started (Initial Local Setup)

Until you finish Task 1 (Docker), you will need to run the app locally on your computer to test your code. 
*(If you run into issues here, it proves exactly why we need Docker!)*

1. You MUST have Python 3.10 or 3.11 installed. (3.12+ will break the AI).
2. You MUST have FFmpeg installed on your Windows system.
3. Double-click the `setup.bat` file. (This downloads ~2.4GB of data, so it may take a while).
4. Double-click `run_all.bat` to launch the app!

Troubleshooting:
- If the app crashes with a `numpy` error, it means your virtual environment is corrupted. Delete the `venv` folder and run `setup.bat` again.
- If you get a missing `gradio` error, your download timed out. Open a terminal in your `venv` and manually run `pip install gradio edge-tts pydub transliterate soundfile`.
- If you get a massive red C++ Build Tools error while it tries to install `TTS`, don't panic. You don't actually need that specific library for the UI to run. Just manually install the other UI packages.

Good luck, and happy coding!
