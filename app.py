# app.py - Improved Live Translation with Smart Audio Processing
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
import uuid
import logging
from typing import Dict, List
import os
from datetime import datetime, timedelta
import base64
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI client
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Live Translation API - Smart Audio Processing", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global storage for rooms and connections
rooms: Dict[str, dict] = {}
connections: Dict[str, List[WebSocket]] = {}

class AudioBuffer:
    """Smart audio buffer with silence detection"""
    
    def __init__(self, room_code: str):
        self.room_code = room_code
        self.audio_chunks = []
        self.last_audio_time = None
        self.is_processing = False
        self.silence_threshold = 3.0  # 3 seconds of silence before processing
        self.max_buffer_time = 10.0   # Max 10 seconds before forced processing
        self.min_buffer_time = 2.0    # Min 2 seconds before considering processing
        self.buffer_start_time = None
        self.consecutive_empty_chunks = 0  # Track empty/silent chunks
        self.max_empty_chunks = 5  # Max empty chunks before clearing buffer
        
    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk to buffer with silence detection"""
        current_time = datetime.now()
        
        # Check if this chunk is likely silence (very small)
        is_likely_silence = len(audio_data) < 1000  # Less than 1KB
        
        if is_likely_silence:
            self.consecutive_empty_chunks += 1
            logger.debug(f"Room {self.room_code}: Detected silent chunk ({self.consecutive_empty_chunks}/{self.max_empty_chunks})")
            
            # If we've had too many silent chunks, clear the buffer
            if self.consecutive_empty_chunks >= self.max_empty_chunks:
                if self.audio_chunks:
                    logger.info(f"Room {self.room_code}: Clearing buffer due to extended silence")
                    self.audio_chunks = []
                    self.buffer_start_time = None
                    self.consecutive_empty_chunks = 0
                return
        else:
            # Reset silence counter on meaningful audio
            self.consecutive_empty_chunks = 0
            
            # Start new buffer if this is the first chunk
            if not self.audio_chunks:
                self.buffer_start_time = current_time
                
            self.audio_chunks.append(audio_data)
            self.last_audio_time = current_time
            
            # Log buffer status
            buffer_duration = (current_time - self.buffer_start_time).total_seconds()
            logger.info(f"Room {self.room_code}: Audio buffer now {len(self.audio_chunks)} chunks, {buffer_duration:.1f}s")
        
    def should_process(self) -> bool:
        """Determine if we should process the current buffer"""
        if not self.audio_chunks or self.is_processing:
            return False
            
        current_time = datetime.now()
        
        # How long since we started buffering
        buffer_duration = (current_time - self.buffer_start_time).total_seconds()
        
        # How long since last audio chunk
        silence_duration = (current_time - self.last_audio_time).total_seconds()
        
        # Must have minimum content to process
        total_audio_size = sum(len(chunk) for chunk in self.audio_chunks)
        if total_audio_size < 5000:  # Less than 5KB total is probably not speech
            return False
        
        # Process if:
        # 1. We have enough audio AND there's been silence for 3+ seconds
        # 2. OR we've been buffering for 10+ seconds (force processing)
        should_process = (
            (buffer_duration >= self.min_buffer_time and silence_duration >= self.silence_threshold) or
            (buffer_duration >= self.max_buffer_time)
        )
        
        if should_process:
            logger.info(f"Room {self.room_code}: Processing buffer - Duration: {buffer_duration:.1f}s, Silence: {silence_duration:.1f}s, Size: {total_audio_size} bytes")
            
        return should_process
        
    def get_audio_for_processing(self) -> bytes:
        """Get combined audio data and reset buffer"""
        if not self.audio_chunks:
            return b""
            
        # Combine all audio chunks
        combined_audio = b"".join(self.audio_chunks)
        
        # Reset buffer
        self.audio_chunks = []
        self.buffer_start_time = None
        self.last_audio_time = None
        self.consecutive_empty_chunks = 0
        
        return combined_audio

class RoomManager:
    def __init__(self):
        self.active_rooms = {}
        self.audio_buffers = {}  # Store audio buffers per room
    
    def create_room(self) -> str:
        """Generate a unique 6-character room code"""
        import random
        import string
        
        while True:
            room_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            if room_code not in self.active_rooms:
                break
        
        self.active_rooms[room_code] = {
            "host": None,
            "audience": [],
            "created_at": datetime.now(),
            "is_active": False
        }
        connections[room_code] = []
        self.audio_buffers[room_code] = AudioBuffer(room_code)
        
        logger.info(f"Created room: {room_code}")
        return room_code
    
    def join_room(self, room_code: str, websocket: WebSocket, user_type: str, language: str = None):
        """Add user to room"""
        if room_code not in self.active_rooms:
            raise HTTPException(status_code=404, detail="Room not found")
        
        connections[room_code].append(websocket)
        
        if user_type == "host":
            self.active_rooms[room_code]["host"] = websocket
            logger.info(f"Host joined room: {room_code}")
        else:
            self.active_rooms[room_code]["audience"].append({
                "websocket": websocket,
                "language": language
            })
            logger.info(f"Audience member joined room: {room_code} (language: {language})")
    
    def remove_from_room(self, room_code: str, websocket: WebSocket):
        """Remove user from room"""
        if room_code in connections:
            if websocket in connections[room_code]:
                connections[room_code].remove(websocket)
        
        if room_code in self.active_rooms:
            self.active_rooms[room_code]["audience"] = [
                user for user in self.active_rooms[room_code]["audience"] 
                if user["websocket"] != websocket
            ]
            
            if self.active_rooms[room_code]["host"] == websocket:
                self.active_rooms[room_code]["host"] = None
                self.active_rooms[room_code]["is_active"] = False
                logger.info(f"Host left room: {room_code}")

room_manager = RoomManager()

class OpenAITranslationService:
    @staticmethod
    async def transcribe_audio(audio_data: bytes) -> str:
        """Convert audio to text using OpenAI Whisper with silence filtering"""
        try:
            # Skip if audio data is too small (likely silence)
            if len(audio_data) < 1000:  # Less than ~1KB is probably silence
                logger.info("Skipping tiny audio chunk (likely silence)")
                return ""
            
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.webm"
            
            # Use more precise Whisper settings
            transcript = await openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="verbose_json",
                temperature=0.1  # Very low temperature for consistency
            )
            
            # Get the text from verbose response
            text = transcript.text.strip()
            
            # Advanced filtering for meaningful speech
            if OpenAITranslationService._is_meaningful_speech(text):
                logger.info(f"✅ Valid transcription ({len(text)} chars): {text}")
                return text
            else:
                logger.info(f"❌ Filtered out: '{text}' (not meaningful speech)")
                return ""
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    @staticmethod
    def _is_meaningful_speech(text: str) -> bool:
        """Check if transcribed text is actually meaningful speech"""
        if not text:
            return False
            
        # Remove punctuation and convert to lowercase
        clean_text = ''.join(c.lower() for c in text if c.isalnum() or c.isspace()).strip()
        
        # Too short
        if len(clean_text) < 3:
            return False
            
        # Common non-speech patterns that Whisper sometimes outputs
        noise_patterns = [
            "thank you", "thanks", "thank", "you", "uh", "um", "hmm", "mm", "ah", 
            "oh", "okay", "ok", "yes", "yeah", "no", "huh", "what", "well",
            "so", "and", "the", "a", "an", "is", "are", "was", "were",
            "music", "applause", "laughter", "silence", "noise", "sound",
            ".", "..", "...", "?", "!", "-", "--"
        ]
        
        # If it's ONLY noise patterns, reject it
        words = clean_text.split()
        if len(words) <= 2 and all(word in noise_patterns for word in words):
            return False
            
        # Reject very repetitive text (like "thank you thank you thank you")
        if len(words) >= 3:
            unique_words = set(words)
            if len(unique_words) <= 2 and len(words) >= 4:  # Too repetitive
                return False
        
        # Check for minimum meaningful length
        if len(clean_text) < 5:
            return False
            
        # Passed all filters - this seems like real speech
        return True
    
    @staticmethod
    async def translate_with_openai(text: str, target_language: str) -> str:
        """Use OpenAI GPT to translate text with better prompts"""
        try:
            # Better language name mapping
            language_names = {
                'ur': 'Urdu', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
                'ko': 'Korean', 'zh': 'Chinese (Simplified)', 'ar': 'Arabic', 'hi': 'Hindi',
                'bn': 'Bengali', 'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch',
                'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish'
            }
            
            target_lang_name = language_names.get(target_language, target_language)
            
            # More precise translation prompt
            prompt = f"""Translate this English text to {target_lang_name}. 
Requirements:
- Provide ONLY the translation, no explanations
- Keep the same tone and meaning
- Make it natural and conversational
- If it's a greeting, use appropriate cultural greeting

Text to translate: "{text}"

Translation:"""
            
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3  # Consistent translations
            )
            
            translation = response.choices[0].message.content.strip()
            
            # Clean up the translation (remove quotes if GPT added them)
            translation = translation.strip('"').strip("'").strip()
            
            logger.info(f"Translated to {target_lang_name}: {translation}")
            return translation
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return f"[{target_language.upper()}] {text}"
    
    @staticmethod
    async def text_to_speech(text: str, language: str = "en") -> bytes:
        """Convert text to speech with optimized settings"""
        try:
            # Better voice mapping for different languages
            voice_mapping = {
                "en": "alloy",     # Clear, professional
                "es": "nova",      # Warm, friendly  
                "fr": "shimmer",   # Elegant
                "de": "onyx",      # Clear, authoritative
                "ur": "alloy",     # Works well with Urdu
                "ar": "fable",     # Good for Arabic
                "hi": "echo",      # Good for Hindi
                "zh": "alloy",     # Works with Chinese
                "ja": "alloy",     # Works with Japanese
                "ko": "alloy",     # Works with Korean
                "pt": "nova",      # Portuguese
                "it": "shimmer",   # Italian
                "ru": "onyx",      # Russian
                "tr": "echo",      # Turkish
                "pl": "onyx",      # Polish
                "nl": "alloy"      # Dutch
            }
            
            selected_voice = voice_mapping.get(language, "alloy")
            
            response = await openai_client.audio.speech.create(
                model="tts-1",  # Fast model for real-time
                voice=selected_voice,
                input=text,
                response_format="mp3",
                speed=1.0  # Normal speed
            )
            
            logger.info(f"Generated TTS for: {text[:30]}... (voice: {selected_voice})")
            return response.content
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""

translation_service = OpenAITranslationService()

# Background task to process audio buffers
async def process_audio_buffers():
    """Background task that checks for audio ready to process"""
    while True:
        try:
            for room_code, audio_buffer in room_manager.audio_buffers.items():
                if audio_buffer.should_process() and not audio_buffer.is_processing:
                    # Mark as processing to prevent duplicate processing
                    audio_buffer.is_processing = True
                    
                    # Get the audio data
                    audio_data = audio_buffer.get_audio_for_processing()
                    
                    if audio_data:
                        # Process in background
                        asyncio.create_task(process_buffered_audio(room_code, audio_data, audio_buffer))
                    else:
                        audio_buffer.is_processing = False
                        
        except Exception as e:
            logger.error(f"Error in audio buffer processing: {e}")
            
        # Check every 500ms for responsive processing
        await asyncio.sleep(0.5)

async def process_buffered_audio(room_code: str, audio_data: bytes, audio_buffer: AudioBuffer):
    """Process accumulated audio data with additional validation"""
    try:
        logger.info(f"Processing buffered audio for room {room_code} ({len(audio_data)} bytes)")
        
        # Double-check audio size before processing
        if len(audio_data) < 5000:  # Less than 5KB is probably not meaningful speech
            logger.info(f"Skipping small audio buffer ({len(audio_data)} bytes) - likely silence")
            return
        
        # Transcribe the audio
        text = await translation_service.transcribe_audio(audio_data)
        
        if text.strip():
            logger.info(f"✅ Broadcasting translation: '{text}'")
            # Broadcast translation to audience
            await broadcast_translation(room_code, text)
        else:
            logger.info(f"❌ No meaningful text transcribed for room {room_code}")
            
    except Exception as e:
        logger.error(f"Error processing buffered audio for room {room_code}: {e}")
    finally:
        # Reset processing flag
        audio_buffer.is_processing = False

# Start background task when app starts
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(process_audio_buffers())
    logger.info("Started audio buffer processing task")

# API Routes
@app.post("/create-room")
async def create_room():
    """Create a new translation room"""
    room_code = room_manager.create_room()
    return {"room_code": room_code, "status": "created"}

@app.get("/room/{room_code}/status")
async def get_room_status(room_code: str):
    """Get room status and participant count"""
    if room_code not in room_manager.active_rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    room = room_manager.active_rooms[room_code]
    return {
        "room_code": room_code,
        "is_active": room["is_active"],
        "host_connected": room["host"] is not None,
        "audience_count": len(room["audience"]),
        "created_at": room["created_at"]
    }

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{room_code}")
async def websocket_endpoint(websocket: WebSocket, room_code: str):
    await websocket.accept()
    
    try:
        data = await websocket.receive_json()
        user_type = data.get("type")
        language = data.get("language")
        
        room_manager.join_room(room_code, websocket, user_type, language)
        
        await websocket.send_json({
            "type": "connected",
            "room_code": room_code,
            "user_type": user_type
        })
        
        while True:
            data = await websocket.receive_json()
            await handle_websocket_message(websocket, room_code, data)
            
    except WebSocketDisconnect:
        room_manager.remove_from_room(room_code, websocket)
        logger.info(f"Client disconnected from room {room_code}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

async def handle_websocket_message(websocket: WebSocket, room_code: str, data: dict):
    """Handle incoming WebSocket messages"""
    message_type = data.get("type")
    
    if message_type == "audio_data":
        # Add audio to buffer instead of processing immediately
        audio_base64 = data.get("audio")
        try:
            audio_bytes = base64.b64decode(audio_base64)
            
            # Add to room's audio buffer
            if room_code in room_manager.audio_buffers:
                room_manager.audio_buffers[room_code].add_audio_chunk(audio_bytes)
                
        except Exception as e:
            logger.error(f"Error buffering audio: {e}")
    
    elif message_type == "start_session":
        if room_code in room_manager.active_rooms:
            room_manager.active_rooms[room_code]["is_active"] = True
            await broadcast_to_room(room_code, {
                "type": "session_started",
                "message": "Host has started the session"
            })
    
    elif message_type == "stop_session":
        if room_code in room_manager.active_rooms:
            room_manager.active_rooms[room_code]["is_active"] = False
            # Clear any remaining audio buffer
            if room_code in room_manager.audio_buffers:
                room_manager.audio_buffers[room_code].audio_chunks = []
            await broadcast_to_room(room_code, {
                "type": "session_stopped", 
                "message": "Host has stopped the session"
            })
    
    elif message_type == "ping":
        await websocket.send_json({"type": "pong"})

async def broadcast_translation(room_code: str, original_text: str):
    """Broadcast translations to all audience members"""
    if room_code not in room_manager.active_rooms:
        return
    
    room = room_manager.active_rooms[room_code]
    translation_tasks = []
    
    for audience_member in room["audience"]:
        websocket = audience_member["websocket"]
        language = audience_member["language"]
        task = process_and_send_translation(websocket, original_text, language)
        translation_tasks.append(task)
    
    if translation_tasks:
        await asyncio.gather(*translation_tasks, return_exceptions=True)

async def process_and_send_translation(websocket: WebSocket, original_text: str, language: str):
    """Process translation and TTS for a single audience member"""
    try:
        # Translate text
        translated_text = await translation_service.translate_with_openai(original_text, language)
        
        # Generate audio
        audio_bytes = await translation_service.text_to_speech(translated_text, language)
        audio_base64 = base64.b64encode(audio_bytes).decode() if audio_bytes else ""
        
        await websocket.send_json({
            "type": "translation",
            "original": original_text,
            "translated": translated_text,
            "language": language,
            "audio": audio_base64
        })
        
    except Exception as e:
        logger.error(f"Error sending translation: {e}")

async def broadcast_to_room(room_code: str, message: dict):
    """Broadcast message to all connections in a room"""
    if room_code in connections:
        disconnected = []
        for websocket in connections[room_code]:
            try:
                await websocket.send_json(message)
            except:
                disconnected.append(websocket)
        
        for ws in disconnected:
            connections[room_code].remove(ws)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "active_rooms": len(room_manager.active_rooms),
        "total_connections": sum(len(conns) for conns in connections.values())
    }

@app.get("/")
async def serve_frontend():
    """Serve the main application"""
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found!")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=sk-your-actual-key-here")
        exit(1)
    
    print("🚀 Starting Live Translation Server v2.0...")
    print("🎯 Smart Audio Processing: Voice detection + sentence completion")
    print("⚡ 2-3 second delay for accurate real-time translation")
    print("🌐 Visit: http://localhost:8000")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")