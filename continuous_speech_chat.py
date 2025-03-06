"""
Continuous Speech Conversation with LLM and Kokoro TTS

This script implements an open-ended conversation where:
1. The user speaks continuously (audio recorded directly with PyAudio)
2. Speech is transcribed using Faster Whisper with CUDA acceleration
3. When the user pauses, the transcription is sent to the LLM
4. The LLM response is converted to speech using Kokoro TTS
5. The conversation continues without needing to press buttons

Requirements:
- pip install kokoro>=0.8.4 soundfile faster-whisper pygame pyaudio numpy
- Mistral running locally via Ollama (or adjust the LLM_API_URL)
"""

import threading
import time
import requests
import json
import os
import soundfile as sf
from kokoro import KPipeline
import queue
from faster_whisper import WhisperModel
import pyaudio
import wave
import numpy as np
import traceback
import pickle
import hashlib
from pathlib import Path
import io
import sounddevice as sd

# Configuration
LLM_API_URL = "http://localhost:11434/api/chat"  # Ollama API endpoint
LLM_MODEL = "mistral"  # Model to use
VOICE = "af_heart"  # Kokoro TTS voice (af_heart, af_bella, etc.)
LANG_CODE = "a"  # 'a' for American English, 'b' for British English
SILENCE_THRESHOLD = 1500  # Amplitude threshold for silence detection
SILENCE_DURATION = 1.5  # Seconds of silence before processing speech
OUTPUT_DIR = "audio_output"  # Directory to store audio files
WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large-v3
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz audio
CHUNK = 1024
CACHE_DIR = "model_cache"  # Directory to store model cache
OLLAMA_CACHE_EXPIRY = 3600  # Cache Ollama responses for 1 hour (in seconds)
MAX_TOKENS_PER_CHUNK = 30  # Maximum tokens to wait for before sending to TTS
TTS_SAMPLE_RATE = 24000  # Sample rate for Kokoro TTS (24kHz)
MAX_SENTENCE_LENGTH = 50  # Maximum words in a sentence before forcing a break
AUDIO_CROSSFADE_MS = 200  # Milliseconds to crossfade between audio chunks
TEXT_BUFFER_SIZE = 3  # Number of sentence chunks to accumulate before processing

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Global variables
pipeline = None
whisper_model = None
models_loaded = threading.Event()
is_processing = threading.Event()  # Flag to indicate when processing a message
ollama_cache = {}  # In-memory cache for Ollama responses
streaming_finished = threading.Event()  # Flag to indicate when streaming is complete
exit_program = threading.Event()  # Flag to signal program should exit

# Pipeline queues for parallel processing
text_chunk_queue = queue.Queue()  # Queue for text chunks from LLM
audio_queue = queue.Queue(maxsize=10)  # Queue for audio chunks to be played (with buffer limit)
audio_playing = threading.Event()  # Flag to indicate when audio is playing

# Message queue for communication between threads
message_queue = queue.Queue()
response_queue = queue.Queue()

# Conversation history for context
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant engaging in a natural conversation. Respond appropriately to the user's messages - if they say something like 'thank you' or 'goodbye', respond with a natural acknowledgment like 'You're welcome' or 'Goodbye', not with a list of capabilities. Keep your responses concise, contextual and conversational."}
]

def get_cache_path(model_name, model_size=None):
    """Generate a cache path for a model"""
    if model_size:
        cache_key = f"{model_name}_{model_size}"
    else:
        cache_key = model_name
    
    # Create a hash of the cache key to use as filename
    hash_obj = hashlib.md5(cache_key.encode())
    cache_hash = hash_obj.hexdigest()
    
    return os.path.join(CACHE_DIR, f"{cache_key}_{cache_hash}.cache")

def load_models():
    """Load models with caching"""
    global pipeline, whisper_model
    
    print("Initializing Kokoro TTS pipeline...")
    pipeline = KPipeline(lang_code=LANG_CODE)
    print("✅ Kokoro TTS initialized")
    
    # Check for cached Whisper model
    whisper_cache_path = get_cache_path("whisper", WHISPER_MODEL_SIZE)
    
    if os.path.exists(whisper_cache_path):
        print(f"Found cached Whisper model configuration at {whisper_cache_path}")
        # We can't directly load the model from cache, but we can use the cached config
        try:
            with open(whisper_cache_path, 'rb') as f:
                cache_info = pickle.load(f)
                print(f"Using cached Whisper model settings: {cache_info}")
        except Exception as e:
            print(f"Error reading cache info: {e}")
    
    print(f"Loading Faster Whisper model ({WHISPER_MODEL_SIZE}) with CUDA...")
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16", download_root=CACHE_DIR)
    print("✅ Faster Whisper model loaded with CUDA acceleration")
    
    # Save model configuration to cache
    try:
        cache_info = {
            "model_size": WHISPER_MODEL_SIZE,
            "device": "cuda",
            "compute_type": "float16",
            "download_root": CACHE_DIR,
            "cached_at": time.time()
        }
        with open(whisper_cache_path, 'wb') as f:
            pickle.dump(cache_info, f)
        print(f"Whisper model configuration cached to {whisper_cache_path}")
    except Exception as e:
        print(f"Error caching Whisper model configuration: {e}")
    
    # Warm up the model with a dummy inference
    print("Warming up Whisper model...")
    dummy_audio = np.zeros((16000,), dtype=np.float32)  # 1 second of silence
    dummy_file = os.path.join(OUTPUT_DIR, "warmup.wav")
    sf.write(dummy_file, dummy_audio, 16000)
    whisper_model.transcribe(dummy_file, beam_size=5)
    os.remove(dummy_file)
    print("✅ Whisper model warmed up")
    
    # Signal that models are loaded
    models_loaded.set()

def get_ollama_cache_key(model, messages):
    """Generate a cache key for Ollama requests"""
    # Create a string representation of the request
    request_str = f"{model}_{json.dumps(messages, sort_keys=True)}"
    # Hash it to create a cache key
    hash_obj = hashlib.md5(request_str.encode())
    return hash_obj.hexdigest()

def is_sentence_boundary(text):
    """Determine if text ends at a natural sentence boundary"""
    # Strip trailing whitespace for accurate detection
    text = text.rstrip()
    
    # Check for empty text
    if not text:
        return False
    
    # Check for sentence-ending punctuation
    if any(text.endswith(p) for p in ['.', '!', '?']):
        return True
        
    # Check for clause-ending punctuation in longer segments
    if len(text.split()) >= 15 and any(text.endswith(p) for p in [':', ';', ',']):
        return True
    
    # Force break for very long text without punctuation
    if len(text.split()) >= MAX_SENTENCE_LENGTH:
        return True
        
    return False

def split_text_into_sentences(text):
    """Split text into natural sentences for better speech flow"""
    sentences = []
    current_sentence = []
    words = text.split()
    
    # Process word by word
    for i, word in enumerate(words):
        current_sentence.append(word)
        current_text = " ".join(current_sentence)
        
        # If we have a sentence boundary, add it to our sentences list
        if is_sentence_boundary(word) or i == len(words) - 1 or len(current_sentence) >= MAX_SENTENCE_LENGTH:
            sentences.append(current_text)
            current_sentence = []
    
    # Add any remaining text
    if current_sentence:
        sentences.append(" ".join(current_sentence))
        
    return sentences

def stream_llm_response(messages):
    """Stream response from Ollama API with natural sentence chunking"""
    try:
        # Prepare the request
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": True  # Enable streaming
        }
        
        print(f"Sending streaming request to LLM at {LLM_API_URL}...")
        
        # Check cache before streaming
        cache_key = get_ollama_cache_key(LLM_MODEL, messages)
        current_time = time.time()
        
        if cache_key in ollama_cache:
            cached_response, timestamp = ollama_cache[cache_key]
            # Check if the cache is still valid
            if current_time - timestamp < OLLAMA_CACHE_EXPIRY:
                print(f"Using cached LLM response (cached {int(current_time - timestamp)} seconds ago)")
                
                # Split cached response into natural sentences
                sentences = split_text_into_sentences(cached_response)
                
                # Add sentences to the text chunk queue for parallel processing
                for sentence in sentences:
                    text_chunk_queue.put(sentence)
                    # Also yield sentence for tracking full response
                    yield sentence
                
                streaming_finished.set()
                return
        
        # For streaming response
        current_sentence = []
        full_response = []
        
        with requests.post(LLM_API_URL, json=payload, stream=True, timeout=30) as response:
            if response.status_code != 200:
                print(f"Error from LLM API: {response.status_code}")
                print(f"Error details: {response.text}")
                error_msg = "Sorry, I couldn't generate a response at the moment."
                text_chunk_queue.put(error_msg)
                yield error_msg
                streaming_finished.set()
                return
                
            for line in response.iter_lines():
                if line:
                    try:
                        line_data = json.loads(line.decode('utf-8'))
                        
                        # Handle different streaming response formats
                        if "message" in line_data:
                            token = line_data.get("message", {}).get("content", "")
                        elif "response" in line_data:
                            token = line_data.get("response", "")
                        else:
                            token = ""
                        
                        if token:
                            full_response.append(token)
                            current_sentence.append(token)
                            current_text = "".join(current_sentence)
                            
                            # Check if we've reached a sentence boundary
                            if is_sentence_boundary(current_text):
                                # Send this chunk for parallel processing
                                sentence = current_text.strip()
                                if sentence:  # Only process non-empty chunks
                                    text_chunk_queue.put(sentence)
                                    yield sentence
                                current_sentence = []
                    
                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON: {line}")
            
            # Process any remaining text
            final_text = "".join(current_sentence).strip()
            if final_text:
                text_chunk_queue.put(final_text)
                yield final_text
        
        # Cache the full response
        full_response_text = "".join(full_response)
        ollama_cache[cache_key] = (full_response_text, current_time)
        
        # Save the cache periodically
        if len(ollama_cache) % 10 == 0:
            try:
                cache_path = get_cache_path("ollama_responses")
                with open(cache_path, 'wb') as f:
                    pickle.dump(ollama_cache, f)
                print(f"Saved {len(ollama_cache)} responses to cache")
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        streaming_finished.set()
        
    except Exception as e:
        print(f"Error in streaming LLM response: {e}")
        traceback.print_exc()
        error_msg = "Sorry, there was an error connecting to the language model."
        text_chunk_queue.put(error_msg)
        yield error_msg
        streaming_finished.set()

def audio_generation_worker():
    """Dedicated worker thread for converting text to speech"""
    print("Starting audio generation worker thread...")
    
    while True:
        try:
            # Get text chunk from queue
            text_chunk = text_chunk_queue.get()
            
            # Check for exit signal
            if text_chunk is None:
                print("Audio generation worker received exit signal")
                audio_queue.put(None)  # Signal playback thread to exit
                break
            
            # Skip empty chunks
            if not text_chunk.strip():
                text_chunk_queue.task_done()
                continue
                
            print(f"Generating audio for: {text_chunk}")
            
            # Generate speech for this chunk
            try:
                generator = pipeline(text_chunk.strip(), voice=VOICE, speed=1.0)
                
                for _, _, audio in generator:
                    # Convert PyTorch Tensor to NumPy array if needed
                    if hasattr(audio, 'detach'):  # Check if it's a PyTorch Tensor
                        audio = audio.detach().cpu().numpy()
                    
                    # Ensure the data is float32
                    if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                    
                    # Normalize if needed
                    max_val = np.max(np.abs(audio))
                    if max_val > 1.0:
                        audio = audio / max_val
                    
                    # Add to audio queue for playback
                    print(f"Adding {len(audio)/TTS_SAMPLE_RATE:.2f}s audio to queue")
                    audio_queue.put(audio)
                    break  # Just use the first chunk
            
            except Exception as e:
                print(f"Error generating speech: {e}")
                traceback.print_exc()
            
            # Mark this text chunk as processed
            text_chunk_queue.task_done()
            
        except Exception as e:
            print(f"Error in audio generation worker: {e}")
            traceback.print_exc()
            
            # Make sure to mark the task as done even in case of error
            try:
                text_chunk_queue.task_done()
            except:
                pass
            
            time.sleep(0.5)  # Wait before trying again

def crossfade_audio(audio1, audio2, crossfade_samples):
    """Blend the end of audio1 with the beginning of audio2 using crossfade"""
    if len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
        # If chunks are too short for crossfade, just concatenate
        return np.concatenate([audio1, audio2])
    
    # Create crossfade weights
    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)
    
    # Apply crossfade
    result = np.concatenate([
        audio1[:-crossfade_samples],
        audio1[-crossfade_samples:] * fade_out + audio2[:crossfade_samples] * fade_in,
        audio2[crossfade_samples:]
    ])
    
    return result

def audio_playback_thread():
    """Thread that plays audio with crossfading between chunks"""
    print("Starting audio playback thread...")
    
    # Buffer to hold multiple chunks for crossfading
    audio_buffer = None
    crossfade_samples = int(AUDIO_CROSSFADE_MS * TTS_SAMPLE_RATE / 1000)
    
    while True:
        try:
            # Get audio data from queue
            audio_data = audio_queue.get()
            
            # Check for exit signal
            if audio_data is None:
                print("Audio playback thread received exit signal")
                break
                
            # Set flag indicating audio is playing
            audio_playing.set()
            
            # First chunk or after a pause
            if audio_buffer is None:
                audio_buffer = audio_data
            else:
                # Crossfade with previous audio
                audio_buffer = crossfade_audio(audio_buffer, audio_data, crossfade_samples)
            
            # Play when we have enough audio or the buffer gets too large
            # This ensures we have a continuous stream to avoid gaps
            if len(audio_buffer) > TTS_SAMPLE_RATE * 1.0:  # Play when buffer has >1s of audio
                # Play the audio
                duration = len(audio_buffer) / TTS_SAMPLE_RATE
                print(f"Playing audio segment ({duration:.2f}s)...")
                
                sd.play(audio_buffer, TTS_SAMPLE_RATE)
                sd.wait()  # Wait for playback to complete
                
                audio_buffer = None  # Reset buffer after playing
                print("Audio segment complete")
            
            # Mark task as done
            audio_queue.task_done()
            
            # Clear flag if queue is empty
            if audio_queue.empty():
                audio_playing.clear()
            
        except Exception as e:
            print(f"Error in audio playback: {e}")
            traceback.print_exc()
            
            # Mark task as done even in case of error
            try:
                audio_queue.task_done()
            except:
                pass
            
            # Reset buffer on error
            audio_buffer = None
            audio_playing.clear()
            time.sleep(0.5)  # Wait before trying again

def say_welcome_message():
    """Say a welcome message when the program starts"""
    welcome_message = "Hello, I'm an AI assistant. How can I help you today?"
    print("Assistant: " + welcome_message)
    
    # Generate and play welcome message
    try:
        generator = pipeline(welcome_message, voice=VOICE, speed=1.0)
        
        for _, _, audio in generator:
            # Convert PyTorch Tensor to NumPy array if needed
            if hasattr(audio, 'detach'):  # Check if it's a PyTorch Tensor
                audio = audio.detach().cpu().numpy()
            
            # Ensure the data is float32
            if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
            
            # Play the welcome audio
            sd.play(audio, TTS_SAMPLE_RATE)
            sd.wait()  # Wait until audio has finished playing
            break
    
    except Exception as e:
        print(f"Error playing welcome message: {e}")
        traceback.print_exc()

def process_messages():
    """Process messages from the queue and coordinate the pipeline stages"""
    # Wait for models to be loaded
    models_loaded.wait()
    
    # Start the worker threads for the pipeline
    audio_gen_thread = threading.Thread(target=audio_generation_worker)
    audio_gen_thread.daemon = True
    audio_gen_thread.start()
    
    audio_play_thread = threading.Thread(target=audio_playback_thread)
    audio_play_thread.daemon = True
    audio_play_thread.start()
    
    while not exit_program.is_set():
        try:
            # Get message from queue with a timeout so we can check exit flag
            try:
                message = message_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if message is None:  # Exit signal
                # Signal all threads to exit
                text_chunk_queue.put(None)
                break
                
            # Add user message to conversation history
            print(f"Processing user message: '{message}'")
            conversation_history.append({"role": "user", "content": message})
            
            # Special handling for common conversational phrases
            simple_responses = {
                "thank you": "You're welcome! Is there anything else I can help you with?",
                "thanks": "You're welcome! Is there anything else I can help you with?",
                "goodbye": "Goodbye! Have a great day.",
                "bye": "Goodbye! Have a great day."
            }
            
            # Check if message is a simple phrase that needs a direct response
            message_lower = message.lower().strip().rstrip('.!?')
            if message_lower in simple_responses and len(message.split()) <= 3:
                # Use direct response for very simple messages
                print(f"Using direct response for '{message}'")
                direct_response = simple_responses[message_lower]
                
                # Add to conversation history
                conversation_history.append({"role": "assistant", "content": direct_response})
                
                # Send directly to text chunk queue
                text_chunk_queue.put(direct_response)
                
                # Clear the processing flag after playback
                text_chunk_queue.join()  # Wait for text processing
                audio_queue.join()  # Wait for all audio to be played
                is_processing.clear()
                
                # Put response in queue to signal completion
                response_queue.put(direct_response)
                
                # If the user said goodbye or bye, signal the program to exit
                if message_lower in ["goodbye", "bye"]:
                    print("User said goodbye, will exit after response...")
                    time.sleep(1)  # Give a moment for the goodbye to be heard
                    exit_program.set()
                
                continue
            
            # Reset streaming finished flag
            streaming_finished.clear()
            
            # Get streaming response from LLM
            print("Starting streaming response...")
            full_response = []
            
            # Collect the full response while pipeline processes in parallel
            for text_chunk in stream_llm_response(conversation_history):
                full_response.append(text_chunk)
                # Note: stream_llm_response already adds chunks to text_chunk_queue
            
            # Wait for streaming to complete
            streaming_finished.wait()
            
            # Combine all chunks for the full response
            response_text = " ".join(full_response).strip()
            
            # Add assistant response to conversation history
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # Wait for all pipeline stages to complete
            print("Waiting for text processing to complete...")
            text_chunk_queue.join()
            
            print("Waiting for audio playback to complete...")
            audio_queue.join()
            
            # Clear the processing flag
            is_processing.clear()
            print("Processing complete")
            
            # Put response in queue to signal completion
            response_queue.put(response_text)
            
        except Exception as e:
            print(f"Error processing message: {e}")
            traceback.print_exc()
            is_processing.clear()  # Make sure to clear the flag in case of error
            response_queue.put("Sorry, I encountered an error processing your message.")

def play_audio(audio_path):
    """Play audio file using sounddevice"""
    if audio_path and os.path.exists(audio_path):
        try:
            print("Playing audio response...")
            
            # Load audio file
            audio_data, sample_rate = sf.read(audio_path, dtype='float32')
            
            # Normalize if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Play the audio
            sd.play(audio_data, sample_rate)
            sd.wait()  # Wait until audio has finished playing
            
            print("Audio playback complete")
                
        except Exception as e:
            print(f"Error playing audio: {e}")
            traceback.print_exc()

def load_ollama_cache():
    """Load the Ollama response cache from disk"""
    global ollama_cache
    
    cache_path = get_cache_path("ollama_responses")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                loaded_cache = pickle.load(f)
                
                # Filter out expired entries
                current_time = time.time()
                valid_entries = {k: v for k, v in loaded_cache.items() 
                               if current_time - v[1] < OLLAMA_CACHE_EXPIRY}
                
                ollama_cache = valid_entries
                print(f"Loaded {len(ollama_cache)} valid responses from cache")
        except Exception as e:
            print(f"Error loading Ollama cache: {e}")
            ollama_cache = {}
    else:
        print("No Ollama cache found, starting with empty cache")
        ollama_cache = {}

def is_silent(data_chunk, threshold=SILENCE_THRESHOLD):
    """Check if the audio chunk is silent"""
    # Convert audio chunk to numpy array
    audio_data = np.frombuffer(data_chunk, dtype=np.int16)
    # Calculate the maximum absolute amplitude
    max_amplitude = np.max(np.abs(audio_data))
    # Return True if the maximum amplitude is below the threshold
    return max_amplitude < threshold

def record_and_transcribe_continuously():
    """Continuously record audio and transcribe it when silence is detected"""
    # Wait for models to be loaded
    print("Waiting for models to load...")
    models_loaded.wait()
    
    p = None
    stream = None
    
    try:
        print("\nInitializing audio recording...")
        p = pyaudio.PyAudio()
        
        # Print available devices for debugging
        print("\nAvailable audio devices:")
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            print(f"Device {i}: {dev_info['name']}")
            print(f"  Input channels: {dev_info['maxInputChannels']}")
        
        # Open the stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print("🎤 Listening... (speak naturally, pause when you're done)")
        print(f"Silence threshold: {SILENCE_THRESHOLD}, Duration: {SILENCE_DURATION}s")
        
        frames = []
        silent_chunks = 0
        recording = False
        max_silent_chunks = int(SILENCE_DURATION * RATE / CHUNK)  # Convert seconds to chunks
        
        # For debugging - count how many consecutive non-silent chunks we've seen
        consecutive_sound_chunks = 0
        
        print("Ready to record speech!")
        
        while not exit_program.is_set():
            try:
                # Don't record while processing a message
                if is_processing.is_set():
                    time.sleep(0.1)
                    continue
                
                # Read audio data
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Check if this chunk is silent
                silent = is_silent(data)
                
                # Debug audio levels periodically
                if not silent:
                    consecutive_sound_chunks += 1
                    if consecutive_sound_chunks >= 3 and not recording:
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        max_amplitude = np.max(np.abs(audio_data))
                        print(f"Detected sound: amplitude {max_amplitude} (threshold: {SILENCE_THRESHOLD})")
                        consecutive_sound_chunks = 0
                else:
                    consecutive_sound_chunks = 0
                
                # If we detect sound and we're not recording, start recording
                if not silent and not recording:
                    recording = True
                    print("Speech detected, recording...")
                    frames = [data]  # Start with this chunk
                    silent_chunks = 0
                
                # If we're recording, add the chunk
                elif recording:
                    frames.append(data)
                    
                    # If this chunk is silent, increment the counter
                    if silent:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0
                    
                    # If we've had enough silent chunks, process the recording
                    if silent_chunks >= max_silent_chunks and len(frames) > max_silent_chunks:
                        print("Silence detected, processing speech...")
                        
                        try:
                            # Set the processing flag to pause recording
                            is_processing.set()
                            
                            # Save the recorded audio to a WAV file
                            audio_file = os.path.join(OUTPUT_DIR, "temp_user.wav")
                            wf = wave.open(audio_file, 'wb')
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(frames))
                            wf.close()
                            
                            # Transcribe the audio
                            segments, info = whisper_model.transcribe(audio_file, beam_size=5)
                            transcription = " ".join([segment.text for segment in segments])
                            
                            if transcription:
                                print(f"You: {transcription}")
                                # Put the transcribed text in the queue for processing
                                message_queue.put(transcription.strip())
                                
                                # Wait for response to be processed before continuing
                                print("Processing your message...")
                                response = response_queue.get()
                                print(f"Assistant: {response}")
                                
                                # Check if program should exit
                                if exit_program.is_set():
                                    break
                            else:
                                print("No speech detected in the recording, listening again...")
                                is_processing.clear()  # Clear the processing flag
                        except Exception as e:
                            print(f"Error processing recording: {e}")
                            traceback.print_exc()
                            is_processing.clear()  # Clear the processing flag in case of error
                        
                        # Reset for next recording
                        recording = False
                        frames = []
                        silent_chunks = 0
            except Exception as e:
                print(f"Error in recording loop: {e}")
                traceback.print_exc()
                time.sleep(0.5)  # Wait a bit before trying again
        
        print("Exiting recording loop...")
    
    except KeyboardInterrupt:
        print("Stopping recording...")
    except Exception as e:
        print(f"Critical error in recording function: {e}")
        traceback.print_exc()
    finally:
        # Clean up resources
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
        
        if p is not None:
            try:
                p.terminate()
            except:
                pass

def main():
    """Main function to run the continuous speech conversation"""
    try:
        # Load Ollama cache
        load_ollama_cache()
        
        # Load models
        print("Loading models...")
        load_models()
        
        # Say welcome message
        say_welcome_message()
        
        # Start the message processing thread
        processing_thread = threading.Thread(target=process_messages)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start the continuous recording and transcription in the main thread
        record_and_transcribe_continuously()
        
        # If we get here, we're exiting
        print("Shutting down...")
        
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Critical error in main function: {e}")
        traceback.print_exc()
    finally:
        # Signal the processing thread to exit
        exit_program.set()
        message_queue.put(None)
        
        # Save Ollama cache before exiting
        try:
            if ollama_cache:
                cache_path = get_cache_path("ollama_responses")
                with open(cache_path, 'wb') as f:
                    pickle.dump(ollama_cache, f)
                print(f"Saved {len(ollama_cache)} responses to cache on exit")
        except Exception as e:
            print(f"Error saving cache on exit: {e}")

if __name__ == "__main__":
    print("Starting continuous speech conversation...")
    print("Press Ctrl+C to exit")
    main()
