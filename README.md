# 🎙️ Live Translation API with Smart Audio Processing

This project is a FastAPI-based backend for **real-time audio translation**, enhanced with **smart silence detection** and **OpenAI's GPT for transcription/translation**.

## 🚀 Features

- 📡 WebSocket support for real-time audio streaming
- 🧠 Smart buffering and silence detection (no unnecessary processing)
- 🔄 Multi-room support for simultaneous connections
- 🤖 OpenAI Whisper/GPT integration for audio transcription or translation
- 🌐 CORS-enabled API, easy to connect with frontends

---

## 🧩 Tech Stack

- FastAPI (WebSocket + HTTP API)
- OpenAI API (Whisper + GPT)
- Asyncio for non-blocking audio processing
- dotenv for secure API key management
- Audio chunk buffering with silence detection

---

## 📂 Project Structure

LiveTranslator/
├── static/
│ └── index.html
├── .env
├── .gitignore
└── app.py


