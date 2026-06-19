# Deepfake Audio Detector

A fully local, Celery-backed WhatsApp bot that detects AI-generated voice notes and audio clips using PyTorch, AASIST, and Wav2Vec2 models.

This guide explains how to spin up the entire infrastructure locally on your own computer.

## Prerequisites
1. **Python 3.10+**
2. **FFmpeg** (`sudo dnf install ffmpeg` or `sudo apt install ffmpeg`)
3. **Ngrok Account** (to expose your local bot to Meta)
4. **Upstash Account** (for a free, serverless Redis queue)
5. **Meta Developer Account** (to use the WhatsApp Business API)

---

## 1. Installation

Clone the repository, create a virtual environment, and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the AASIST model weights (if you haven't already):
```bash
git clone https://github.com/clovaai/aasist /tmp/aasist
cp /tmp/aasist/models/weights/AASIST.pth ./weights/AASIST.pth
cp /tmp/aasist/models/AASIST.py ./models/AASIST.py
```

---

## 2. Configuration & Secrets

Copy the example environment file to `.env`:
```bash
cp .env.example .env
```

Open `.env` and fill in the following three critical values:

1. **`WA_TOKEN`**: Go to your Meta Developer Dashboard -> WhatsApp -> API Setup. Click **Generate access token**. *(Note: This token expires every 24 hours. If your bot stops replying with `HTTP 401 Unauthorized`, you need to generate a new one and paste it here).*
2. **`PHONE_NUMBER_ID`**: Found on the same Meta API Setup page.
3. **`REDIS_URL`**: Go to Upstash.com, create a free Redis database, copy the URL, and ensure it begins with `rediss://`.

---

## 3. Running the Infrastructure (The 3 Terminals)

Because this bot is built for heavy AI processing, it is split into a web server (FastAPI) and a background worker (Celery). You need **three separate terminal windows** open simultaneously.

### Terminal 1: The FastAPI Web Server
This server catches incoming messages from Meta and instantly puts them into the Upstash Redis queue.
```bash
source .venv/bin/activate
uvicorn webhook.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

### Terminal 2: The Celery Worker
This worker pulls messages out of Redis, downloads the audio, runs the PyTorch models, and sends the final verdict back to WhatsApp.
```bash
source .venv/bin/activate
celery -A processor.tasks worker --loglevel=info
```

### Terminal 3: The Ngrok Tunnel
This exposes your local Port 8000 to the public internet securely so Meta can reach it.
```bash
ngrok http 8000
```

---

## 4. Connecting Meta to your Bot

Once your 3 terminals are running, look at Terminal 3 (Ngrok) and copy the `Forwarding` URL (it looks like `https://random-words.ngrok-free.dev`).

1. **Bypass Ngrok Warning:** Open a web browser and paste your Ngrok URL. Click the blue **"Visit Site"** button. *(If you skip this, Meta will fail to verify the webhook).*
2. Go to the Meta Developer Dashboard -> WhatsApp -> **Configuration**.
3. Under the Webhook section, click **Edit**.
4. **Callback URL:** Paste your Ngrok URL and append `/webhook` to the end. (e.g., `https://random-words.ngrok-free.dev/webhook`).
5. **Verify Token:** Type in the exact `VERIFY_TOKEN` you set in your `.env` file.
6. Click **Verify and Save**.
7. Directly below that, in the "Webhook fields" section, click **Manage** and subscribe to **`messages`**.

---

## 5. Usage
You are completely live! Message your Meta Test Phone Number via WhatsApp.
* You can record a **Voice Note**.
* You can attach a `.wav` or `.ogg` file as a **Document**.

The bot will automatically download, convert, analyze, and reply with a trust score.
