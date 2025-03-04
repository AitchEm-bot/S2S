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
import pygame
import queue
from faster_whisper import WhisperModel
import pyaudio
import wave
import numpy as np
import traceback
import pickle
import hashlib
from pathlib import Path

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

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Global variables
pipeline = None
whisper_model = None
models_loaded = threading.Event()
is_processing = threading.Event()  # Flag to indicate when processing a message
ollama_cache = {}  # In-memory cache for Ollama responses

# Initialize pygame for audio playback
pygame.mixer.init()

# Message queue for communication between threads
message_queue = queue.Queue()
response_queue = queue.Queue()

# Conversation history for context
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant. Keep your responses concise and conversational."}
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

def get_llm_response(message):
    """Get response from LLM API with caching"""
    try:
        # Prepare the request
        payload = {
            "model": LLM_MODEL,
            "messages": conversation_history,
            "stream": False
        }
        
        # Check if we have a cached response
        cache_key = get_ollama_cache_key(LLM_MODEL, conversation_history)
        current_time = time.time()
        
        if cache_key in ollama_cache:
            cached_response, timestamp = ollama_cache[cache_key]
            # Check if the cache is still valid
            if current_time - timestamp < OLLAMA_CACHE_EXPIRY:
                print(f"Using cached LLM response (cached {int(current_time - timestamp)} seconds ago)")
                return cached_response
            else:
                print("Cache expired, fetching new response")
        
        # Send request to LLM API
        print(f"Sending request to LLM at {LLM_API_URL}...")
        
        # Add timeout to prevent hanging
        response = requests.post(LLM_API_URL, json=payload, timeout=30)
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            # Extract the response text
            response_data = response.json()
            print(f"Response data keys: {response_data.keys()}")
            
            # Handle different API response formats
            if "message" in response_data:
                response_text = response_data["message"]["content"]
            elif "response" in response_data:
                response_text = response_data["response"]
            else:
                print(f"Unexpected response format: {response_data}")
                return "Sorry, I received an unexpected response format from the language model."
            
            # Cache the response
            ollama_cache[cache_key] = (response_text, current_time)
            
            # Save the cache to disk periodically
            if len(ollama_cache) % 10 == 0:  # Save every 10 new responses
                try:
                    cache_path = get_cache_path("ollama_responses")
                    with open(cache_path, 'wb') as f:
                        pickle.dump(ollama_cache, f)
                    print(f"Saved {len(ollama_cache)} responses to cache")
                except Exception as e:
                    print(f"Error saving cache: {e}")
            
            return response_text
        else:
            print(f"Error from LLM API: {response.status_code}")
            try:
                print(f"Error details: {response.text}")
            except:
                pass
            return "Sorry, I couldn't generate a response at the moment."
    
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        traceback.print_exc()
        return "Sorry, there was an error connecting to the language model."

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
        
        while True:
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

def process_messages():
    """Process messages from the queue and generate responses"""
    # Wait for models to be loaded
    models_loaded.wait()
    
    while True:
        try:
            # Get message from queue
            message = message_queue.get()
            if message is None:  # Exit signal
                break
                
            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": message})
            
            # Get response from LLM
            response_text = get_llm_response(message)
            
            # Add assistant response to conversation history
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # Generate speech from response
            audio_path = text_to_speech(response_text)
            
            # Play the audio
            if audio_path:
                play_audio(audio_path)
            
            # Clear the processing flag after playback
            is_processing.clear()
            
            # Put response in queue to signal completion
            response_queue.put(response_text)
            
        except Exception as e:
            print(f"Error processing message: {e}")
            traceback.print_exc()
            is_processing.clear()  # Make sure to clear the flag in case of error
            response_queue.put("Sorry, I encountered an error processing your message.")

def text_to_speech(text):
    """Convert text to speech using Kokoro TTS"""
    try:
        # Use a single temporary file that gets overwritten each time
        output_file = os.path.join(OUTPUT_DIR, "temp_assistant.wav")
        
        # If the file exists, make sure it's not in use
        if os.path.exists(output_file):
            try:
                # Try to open and close the file to check if it's accessible
                with open(output_file, 'a'):
                    pass
            except PermissionError:
                # If file is locked, use an alternative filename
                print("Audio file is locked, using alternative file")
                output_file = os.path.join(OUTPUT_DIR, "temp_alt.wav")
        
        # Special handling for structured text like numbered lists
        if '\n' in text and any(line.strip().startswith(str(i) + '.') for i in range(1, 10) for line in text.split('\n')):
            print("Detected structured text format (numbered list), processing in chunks...")
            
            # For numbered lists, we'll process each item separately but maintain context
            lines = text.split('\n')
            all_audio = []
            
            # Group the lines into chunks of 2-3 items each to maintain context
            # but avoid exceeding Kokoro's processing limits
            chunks = []
            current_chunk = []
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Add line to current chunk
                current_chunk.append(line)
                
                # If this line starts a new numbered item (except for the first one)
                # and we already have 2 or more items, start a new chunk
                if (line.strip()[0].isdigit() and 
                    line.strip()[1:].startswith('.') and 
                    len(current_chunk) > 2 and
                    not line.strip().startswith('1.')):
                    chunks.append('\n'.join(current_chunk[:-1]))
                    current_chunk = [line]
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} of the numbered list...")
                
                # Replace newlines with spaces for better flow in speech
                processed_chunk = chunk.replace('\n\n', '. ').replace('\n', '. ')
                
                # Generate speech for this chunk
                generator = pipeline(processed_chunk, voice=VOICE, speed=1.0)
                
                # Get the audio
                for _, _, audio in generator:
                    all_audio.append(audio)
                    break  # Just use the first chunk
            
            # Combine all audio segments
            if all_audio:
                combined_audio = np.concatenate(all_audio)
                # Save the combined audio to a file
                try:
                    sf.write(output_file, combined_audio, 24000)  # 24kHz sample rate
                    return output_file
                except Exception as e:
                    print(f"Error writing to {output_file}: {e}")
                    # Try alternative filename as last resort
                    alt_file = os.path.join(OUTPUT_DIR, f"response_{int(time.time())}.wav")
                    sf.write(alt_file, combined_audio, 24000)
                    return alt_file
            else:
                print("No audio was generated")
                return None
                
        else:
            # For regular text, process paragraph by paragraph
            # Split text into paragraphs and process each one
            paragraphs = text.split('\n\n')
            all_audio = []
            
            for i, paragraph in enumerate(paragraphs):
                # Skip empty paragraphs
                if not paragraph.strip():
                    continue
                    
                print(f"Processing paragraph {i+1}/{len(paragraphs)}...")
                
                # Process each paragraph
                generator = pipeline(paragraph.strip(), voice=VOICE, speed=1.0)
                
                # Collect audio from generator
                for _, _, audio in generator:
                    all_audio.append(audio)
                    break  # Just use the first chunk for each paragraph
            
            # Combine all audio segments
            if all_audio:
                combined_audio = np.concatenate(all_audio)
                # Save the combined audio to a file
                try:
                    sf.write(output_file, combined_audio, 24000)  # 24kHz sample rate
                    return output_file
                except Exception as e:
                    print(f"Error writing to {output_file}: {e}")
                    # Try alternative filename as last resort
                    alt_file = os.path.join(OUTPUT_DIR, f"response_{int(time.time())}.wav")
                    sf.write(alt_file, combined_audio, 24000)
                    return alt_file
            else:
                print("No audio was generated")
                return None
    
    except Exception as e:
        print(f"Error generating speech: {e}")
        traceback.print_exc()
        return None

def play_audio(audio_path):
    """Play audio file using pygame"""
    if audio_path and os.path.exists(audio_path):
        try:
            print("Playing audio response...")
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            print("Audio playback complete")
            
            # Explicitly unload the music to release the file
            pygame.mixer.music.unload()
            
            # Small delay to ensure file is released
            time.sleep(0.1)
                
        except Exception as e:
            print(f"Error playing audio: {e}")

def main():
    """Main function to run the continuous speech conversation"""
    try:
        # Load Ollama cache
        load_ollama_cache()
        
        # Load models
        print("Loading models...")
        load_models()
        
        # Start the message processing thread
        processing_thread = threading.Thread(target=process_messages)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start the continuous recording and transcription in the main thread
        record_and_transcribe_continuously()
        
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Critical error in main function: {e}")
        traceback.print_exc()
    finally:
        # Signal the processing thread to exit
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
