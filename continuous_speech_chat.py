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
import uuid
import warnings
import re

# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

# Monkey patch torch.nn.utils.weight_norm to use the new parametrizations version
try:
    import torch
    from torch.nn.utils import weight_norm
    
    # Store the original function
    original_weight_norm = weight_norm
    
    # Create a patched version that uses the new parametrizations
    def patched_weight_norm(*args, **kwargs):
        # Suppress the warning by using the new version if available
        try:
            from torch.nn.utils.parametrizations import weight_norm as new_weight_norm
            return new_weight_norm(*args, **kwargs)
        except ImportError:
            # Fall back to the original if the new one isn't available
            return original_weight_norm(*args, **kwargs)
    
    # Apply the patch
    torch.nn.utils.weight_norm = patched_weight_norm
except ImportError:
    # If torch isn't available, just continue
    pass

# Configuration
LLM_API_URL = "http://localhost:11434"  # Ollama API endpoint (base URL)
LLM_MODEL = "mistral"  # Model to use
VOICE = "af_heart"  # Kokoro TTS voice (af_heart, af_bella, etc.)
LANG_CODE = "a"  # 'a' for American English, 'b' for British English
SILENCE_THRESHOLD = 1500  # Amplitude threshold for silence detection
SILENCE_DURATION = 1.5  # Seconds of silence before processing speech
STORAGE_SILENCE_DURATION = 4.5  # Extra silence duration for storage mode (3s more than regular)
STORAGE_TIMEOUT = 30  # Seconds to wait before timing out storage mode
DEBUG_STORAGE = True  # Print debug messages for storage flow
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
STORE_COMMANDS = ["store", "remember this", "save this"]  # Commands to trigger storage mode
USER_TURN_START_SOUND = "audio_cues/listening_audio.wav"  # Replace with your file path
USER_TURN_END_SOUND = "audio_cues/processing_audio.wav"      # Replace with your file path
EXIT_COMMANDS = ["goodbye", "bye", "exit", "quit", "end conversation", "see you later"]  # Commands to exit the program

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
waiting_for_storage = threading.Event()  # Flag to indicate we're waiting for content to store
storage_start_time = None

# Initialize RAG handler if available
try:
    from rag_handler import RAGHandler
    rag_available = True
    print("RAG functionality available")
    rag_handler = RAGHandler()
except ImportError:
    rag_available = False
    print("RAG functionality not available - continuing without memory features")

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

def get_relevant_context(query, max_results=3, use_cache=True):
    """Get relevant context from RAG handler if available
    
    Args:
        query (str): The query to search for relevant context
        max_results (int): Maximum number of results to return
        use_cache (bool): Whether to use cached results
        
    Returns:
        str or None: Relevant context as a formatted string, or None if no context found
    """
    if not rag_available or rag_handler is None:
        return None
    
    try:
        # Search for relevant context using the cached retrieval if enabled
        if use_cache:
            # This will use the LRU cache if available
            context = rag_handler.search_relevant_context(query, max_results=max_results)
        else:
            # Force a new search without using cache
            context = rag_handler._search_relevant_context_impl(query, max_results=max_results)
            
        if context and isinstance(context, str) and len(context) > 0:
            # Format is already correct - just return it
            return context
        elif context and isinstance(context, list) and len(context) > 0:
            # Handle list format (older version compatibility)
            context_items = []
            for item in context:
                if isinstance(item, tuple) and len(item) >= 2:
                    # Handle tuple format (score, text)
                    context_items.append(f"Reference: {item[1]}")
                elif isinstance(item, str):
                    # Handle string format
                    context_items.append(f"Reference: {item}")
            
            if context_items:
                return "\n\n".join(context_items)
        
        return None
    except Exception as e:
        print(f"Error retrieving context: {e}")
        traceback.print_exc()
        return None

def store_to_memory(text, importance=0.7):
    """Store text to memory using RAG handler"""
    if not rag_available or rag_handler is None:
        print("Memory storage not available - skipping")
        return False
    
    try:
        # Store the text in memory
        if DEBUG_STORAGE:
            print(f"📝 Storing to memory: '{text}'")
        else:
            print(f"Storing to memory: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Make sure the text is in a good format for storage
        cleaned_text = text.strip()
        if not cleaned_text or len(cleaned_text) < 3:
            print("Text too short or empty - skipping storage")
            return False
            
        # Try to store message using the RAG handler
        try:
            rag_handler.store_message(cleaned_text, role="user", importance=importance)
            if DEBUG_STORAGE:
                print("✅ Memory storage successful")
            return True
        except Exception as e:
            print(f"Error using store_message, trying alternate method: {e}")
            # Fallback to direct storage method
            try:
                # Get embedding for the text
                embedding = rag_handler.embed_model.encode(cleaned_text)
                
                # Create metadata
                metadata = {
                    "source": "user_input",
                    "timestamp": time.time(),
                    "importance": importance
                }
                
                # Generate a unique ID
                doc_id = str(uuid.uuid4())
                
                # Choose memory store based on importance
                memory_store = rag_handler.short_term_memory
                if importance > 0.7:  # High importance goes to long-term
                    memory_store = rag_handler.long_term_memory
                
                # Store directly in memory
                memory_store.add(
                    documents=[cleaned_text],
                    embeddings=[embedding.tolist()],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                
                if DEBUG_STORAGE:
                    print("✅ Memory storage successful (using fallback method)")
                return True
            except Exception as e2:
                print(f"Error storing to memory (fallback failed): {e2}")
                traceback.print_exc()
                return False
    except Exception as e:
        print(f"Error storing to memory: {e}")
        traceback.print_exc()
        return False

def speak_storage_confirmation():
    """Speak a confirmation message after storing information"""
    confirmation = "Information stored in memory. What else can I help you with?"
    print("Assistant: " + confirmation)
    
    if DEBUG_STORAGE:
        print("🟢 STORAGE COMPLETE - Information stored successfully")
    
    try:
        # Add to standard output pipeline to ensure visibility
        text_chunk_queue.put(confirmation)
        
        # Wait for the audio to be processed and played
        time.sleep(0.5)  # Short delay to ensure the text is queued
        
        # Wait for audio queue to be empty before continuing
        while not audio_queue.empty() or audio_playing.is_set():
            time.sleep(0.1)
    
    except Exception as e:
        print(f"Error playing confirmation: {e}")
        traceback.print_exc()

def speak_storage_prompt():
    """Speak a prompt asking what the user wants to store"""
    prompt = "What would you like me to remember? I'll wait for you to finish speaking."
    print("Assistant: " + prompt)
    
    if DEBUG_STORAGE:
        print("🔴 STORAGE MODE ACTIVATED - Waiting for content to store")
    
    try:
        # Add to standard output pipeline to ensure visibility
        # This will route through the audio_generation_worker and audio_playback_thread
        # instead of playing directly
        text_chunk_queue.put(prompt)
        
        # Wait for the audio to be processed and played
        # This ensures we don't continue until the prompt is spoken
        time.sleep(0.5)  # Short delay to ensure the text is queued
        
        # Wait for audio queue to be empty before continuing
        while not audio_queue.empty() or audio_playing.is_set():
            time.sleep(0.1)
    
    except Exception as e:
        print(f"Error playing storage prompt: {e}")
        traceback.print_exc()

def restart_audio_stream(p=None, stream=None):
    """Safely close and restart the audio stream if needed"""
    try:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        
        if p is None:
            p = pyaudio.PyAudio()
        
        # Open a new stream
        stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
        print("🔄 Audio stream restarted")
        return p, stream
    except Exception as e:
        print(f"Error restarting audio stream: {e}")
        traceback.print_exc()
        return p, None

def load_models():
    """Load models with caching"""
    global pipeline, whisper_model
    
    print("Initializing Kokoro TTS pipeline...")
    # Initialize with custom parameters to avoid warnings
    # Set num_layers=2 when dropout is used to avoid the warning
    pipeline = KPipeline(
        lang_code=LANG_CODE,
        # Pass repo_id explicitly to avoid the warning
        repo_id='hexgrad/Kokoro-82M'
    )
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
    """Generate a cache key for Ollama requests that is consistent for similar queries"""
    try:
        # Extract the last user message
        user_message = ""
        if messages:
            # Find the last user message
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get('role') == 'user' and 'content' in msg:
                    user_message = msg['content']
                    break
        
        if not user_message:
            # No user message found, use a fallback
            return f"no_user_message_{int(time.time())}"
        
        # Normalize the message to ensure consistent keys
        # 1. Convert to lowercase
        normalized = user_message.lower()
        # 2. Remove all punctuation and special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # 3. Remove extra whitespace
        normalized = ' '.join(normalized.split())
        # 4. Truncate to first 10 words to focus on the core question
        words = normalized.split()
        if len(words) > 10:
            normalized = ' '.join(words[:10])
            
        # Create a simple key based on the model and normalized message
        key_base = f"{model}_{normalized}"
        
        # Hash the key base to create a consistent cache key
        hash_obj = hashlib.md5(key_base.encode())
        cache_key = hash_obj.hexdigest()
        
        # Print debug info
        print(f"Original message: '{user_message}'")
        print(f"Normalized message: '{normalized}'")
        print(f"Key base: '{key_base}'")
        
        return cache_key
    except Exception as e:
        print(f"Error generating cache key: {e}")
        traceback.print_exc()
        # Return a unique key to avoid cache hits in case of error
        return f"error_{int(time.time())}"

def is_sentence_boundary(text):
    """Determine if text ends at a natural sentence boundary"""
    # Handle invalid input
    if not text or not isinstance(text, str):
        return False
        
    # Strip trailing whitespace for accurate detection
    text = text.rstrip()
    
    # Check for empty text
    if not text:
        return False
    
    # Check for sentence-ending punctuation with possible closing quotes/brackets
    ending_chars = ['.', '!', '?', '."', '!"', '?"', '."', '!"', '?"', '.)', '!)', '?)', '.")', '!")', '?")']
    if len(text.split()) >= 10 and any(text.endswith(p) for p in ending_chars):
        return True
    
    # Check for dialog ending with quotes
    if text.endswith('"') and text.count('"') % 2 == 0 and len(text) > 10:
        return True
        
    # Check for clause-ending punctuation in longer segments
    if len(text.split()) >= 10 and any(text.endswith(p) for p in [':', ';', ',', '—', '–']):
        return True
    
    # Force break for very long text without punctuation
    if len(text.split()) >= MAX_SENTENCE_LENGTH:
        return True
        
    # Check for common sentence-ending patterns
    common_endings = [
        " however", " nevertheless", " therefore", " thus", " hence", 
        " accordingly", " consequently", " finally", " lastly"
    ]
    for ending in common_endings:
        if text.lower().endswith(ending):
            return True
        
    return False

def split_text_into_sentences(text):
    """Split text into natural sentences for better speech flow"""
    if not text or not isinstance(text, str):
        print("Warning: Invalid text passed to split_text_into_sentences")
        return []
        
    # Clean the text first
    text = text.strip()
    if not text:
        return []
        
    sentences = []
    current_sentence = []
    words = text.split()
    
    if not words:
        return [text]  # Return the original text if no words found
    
    # Process word by word
    for i, word in enumerate(words):
        current_sentence.append(word)
        current_text = " ".join(current_sentence)
        
        # If we have a sentence boundary, add it to our sentences list
        if is_sentence_boundary(current_text) or i == len(words) - 1 or len(current_sentence) >= MAX_SENTENCE_LENGTH:
            if current_text.strip():  # Only add non-empty sentences
                sentences.append(current_text.strip())
            current_sentence = []
    
    # Add any remaining text
    if current_sentence:
        remaining_text = " ".join(current_sentence).strip()
        if remaining_text:  # Only add non-empty text
            sentences.append(remaining_text)
    
    # Final check to ensure we have at least one sentence
    if not sentences and text.strip():
        sentences = [text.strip()]
        
    # Debug info
    print(f"Split text into {len(sentences)} sentences")
        
    return sentences

def stream_llm_response(messages):
    """Stream response from Ollama API with natural sentence chunking"""
    try:
        # Start background retrieval if this is a user message
        background_retrieval_thread = None
        if len(messages) >= 2 and messages[-1]['role'] == 'user':
            user_query = messages[-1]['content']
            
            # Start background retrieval
            if rag_available and rag_handler is not None:
                print("Starting background retrieval while generating response...")
                background_retrieval_thread = rag_handler.start_background_retrieval(user_query)
        
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
        
        print(f"Generated cache key: {cache_key}")
        print(f"Cache has {len(ollama_cache)} entries")
        
        # Print all cache keys for comparison
        if ollama_cache:
            print("Available cache keys:")
            for i, k in enumerate(ollama_cache.keys()):
                print(f"  {i+1}. {k}")
        
        # Check if the key exists in the cache
        if cache_key in ollama_cache:
            print(f"Cache hit for key: {cache_key}")
            cached_response, timestamp = ollama_cache[cache_key]
            cache_age = int(current_time - timestamp)
            
            # Check if the cache is still valid and not empty
            if cache_age < OLLAMA_CACHE_EXPIRY and cached_response and cached_response.strip():
                print(f"Using cached LLM response (cached {cache_age} seconds ago)")
                print(f"Cached response: '{cached_response[:100]}...'")
                
                # Split cached response into natural sentences
                sentences = split_text_into_sentences(cached_response)
                
                if not sentences:
                    print("Warning: Cached response couldn't be split into sentences, using as is")
                    sentences = [cached_response]
                
                print(f"Split into {len(sentences)} sentences for processing")
                
                # Add sentences to the text chunk queue for parallel processing
                for sentence in sentences:
                    if sentence and sentence.strip():
                        text_chunk_queue.put(sentence.strip())
                    # Also yield sentence for tracking full response
                        yield sentence.strip()
                
                streaming_finished.set()
                
                # Check if background retrieval found anything
                if background_retrieval_thread and rag_handler:
                    relevant_context = rag_handler.get_background_retrieval_result(timeout=1.0)
                    if relevant_context:
                        print(f"Background retrieval found relevant context, but using cached response")
                
                return
            else:
                if not cached_response or not cached_response.strip():
                    print(f"Cached response is empty, generating new response")
                else:
                    print(f"Cached response expired (age: {cache_age}s, expiry: {OLLAMA_CACHE_EXPIRY}s), generating new response")
        else:
            print(f"No cache entry found for key: {cache_key}")
        
        # For streaming response
        current_sentence = []
        full_response = []
        
        # Make the streaming request
        response = requests.post(
            f"{LLM_API_URL}/api/chat",  # Correctly append the API endpoint path
            json=payload,
            stream=True,
            timeout=60
        )
        
        # Check for HTTP errors
        if response.status_code != 200:
            error_msg = f"Error from LLM API: HTTP {response.status_code}"
            print(error_msg)
            try:
                error_details = response.text
                print(f"Error details: {error_details}")
            except:
                pass
                
            # Return a fallback response
            fallback_response = "I'm sorry, I'm having trouble connecting to my language model right now. Please try again in a moment."
            text_chunk_queue.put(fallback_response)
            yield fallback_response
            streaming_finished.set()
            return
                
        # Process the streaming response
        for line in response.iter_lines():
            if line:
                try:
                    # Parse the JSON line
                    chunk = json.loads(line)
                    
                    # Check if this is the done message
                    if chunk.get("done", False):
                        # Process any remaining text in current_sentence
                        if current_sentence:
                            final_text = "".join(current_sentence).strip()
                            if final_text:
                                text_chunk_queue.put(final_text)
                                yield final_text
                                full_response.append(final_text)
                        break
                    
                    # Get the text chunk
                    if "message" in chunk and "content" in chunk["message"]:
                        text_chunk = chunk["message"]["content"]
                        
                        # Add to current sentence
                        current_sentence.append(text_chunk)
                        
                        # Check if we have a sentence boundary
                        text_so_far = "".join(current_sentence)
                        
                        if is_sentence_boundary(text_so_far):
                            # We have a complete sentence, add it to the queue
                            text_chunk_queue.put(text_so_far)
                            # Also yield the sentence for tracking full response
                            yield text_so_far
                            full_response.append(text_so_far)
                            # Reset current sentence
                            current_sentence = []
                    
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")
                    continue
        
        # Signal that streaming is finished
        streaming_finished.set()
        
        # Cache the full response
        full_response_text = "".join(full_response)
        
        # Only cache non-empty responses
        if full_response_text and full_response_text.strip():
            print(f"Caching response: '{full_response_text[:100]}...'")
            ollama_cache[cache_key] = (full_response_text, current_time)
            
            # Save the cache immediately after each response
            save_ollama_cache()
        else:
            print("Response is empty, not caching")
        
        # Check if background retrieval found anything
        if background_retrieval_thread and rag_handler:
            relevant_context = rag_handler.get_background_retrieval_result(timeout=1.0)
            if relevant_context:
                print(f"Background retrieval found relevant context after response generation")
                print(f"For future queries on this topic, context will be available immediately")
        
    except Exception as e:
        print(f"Error in stream_llm_response: {e}")
        traceback.print_exc()
        streaming_finished.set()  # Make sure to signal that streaming is finished
        yield f"I encountered an error: {str(e)}"

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
            if not text_chunk or not isinstance(text_chunk, str) or not text_chunk.strip():
                print("Skipping empty or invalid text chunk")
                text_chunk_queue.task_done()
                continue
                
            print(f"Generating audio for: '{text_chunk}'")
            
            # Generate speech for this chunk
            try:
                generator = pipeline(text_chunk.strip(), voice=VOICE, speed=1.0)
                
                audio_generated = False
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
                    print(f"Adding {len(audio)/TTS_SAMPLE_RATE:.2f}s audio to queue for: '{text_chunk[:30]}...'")
                    audio_queue.put(audio)
                    audio_generated = True
                    break  # Just use the first chunk
                
                if not audio_generated:
                    print(f"Warning: No audio generated for text: '{text_chunk}'")
            
            except Exception as e:
                print(f"Error generating speech for '{text_chunk}': {e}")
                traceback.print_exc()
            
            # Mark this text chunk as processed
            text_chunk_queue.task_done()
            
        except Exception as e:
            print(f"Error in audio generation worker: {e}")
            traceback.print_exc()
            # Continue processing other chunks
            try:
                text_chunk_queue.task_done()
            except:
                pass

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
    # Wait for models to be loaded
    print("Waiting for models to load before saying welcome message...")
    models_loaded.wait()
    
    welcome_message = "Hello, I'm an AI assistant. How can I help you today?"
    print("Assistant: " + welcome_message)
    
    try:
        # Generate audio directly without using the pipeline
        print(f"Generating welcome message audio...")
        generator = pipeline(welcome_message.strip(), voice=VOICE, speed=1.0)
        
        # Process the generated audio
        welcome_audio = None
        for _, _, audio in generator:
            # Convert PyTorch Tensor to NumPy array if needed
            if hasattr(audio, 'detach'):
                audio = audio.detach().cpu().numpy()
            
            # Ensure the data is float32
            if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
            
            # Store the audio
            welcome_audio = audio
            break  # We only need the first chunk
        
        if welcome_audio is not None:
            # Play the audio directly
            print("Playing welcome message...")
            sd.play(welcome_audio, TTS_SAMPLE_RATE)
            sd.wait()  # Wait for playback to finish
            print("Welcome message complete")
            
            # Play the user turn start cue
            play_user_turn_start_cue()
        else:
            print("Failed to generate welcome message audio")
    
    except Exception as e:
        print(f"Error playing welcome message: {e}")
        traceback.print_exc()

def process_messages():
    """Process messages from the queue and coordinate the pipeline stages"""
    global storage_start_time
    
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
            
            # Check if we're in storage mode (waiting for content to store)
            if waiting_for_storage.is_set():
                print(f"Storing message to memory: '{message}'")
                # Store the message to memory
                if store_to_memory(message):
                    # Speak a confirmation
                    speak_storage_confirmation()
                    # Add confirmation to response queue to signal completion
                    response_queue.put("Information stored in memory.")
                else:
                    # If storage failed, provide feedback
                    failure_message = "Sorry, I was unable to store that information. Please try again."
                    text_chunk_queue.put(failure_message)
                    response_queue.put(failure_message)
                
                # Clear the storage mode flag - to be fully safe, we do this last
                waiting_for_storage.clear()
                if DEBUG_STORAGE:
                    print("🟡 STORAGE MODE DEACTIVATED - Returning to normal conversation")
                storage_start_time = None  # Reset the storage timeout
                # The recording function now handles clearing the processing flag
                # Don't clear is_processing here to avoid race conditions
                message_queue.task_done()
                continue
            
            # Add user message to conversation history
            print(f"Processing user message: '{message}'")
            
            # Check if this is an exit command
            message_lower = message.lower().strip()
            if any(cmd in message_lower for cmd in EXIT_COMMANDS):
                print("Exit command detected")
                # Generate a goodbye response
                goodbye_message = "Goodbye! It was nice talking with you."
                print("Assistant: " + goodbye_message)
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": message})
                conversation_history.append({"role": "assistant", "content": goodbye_message})
                
                # Play the goodbye message
                text_chunk_queue.put(goodbye_message)
                
                # Wait for audio to finish playing
                print("Waiting for text processing to complete...")
                text_chunk_queue.join()
                
                print("Waiting for audio playback to complete...")
                audio_queue.join()
                
                # Add a 4-second delay after the assistant's response and before prompting the user
                print("Waiting 1 second before prompting user...")
                time.sleep(1.0)
                
                # Play the user turn start cue to indicate it's the user's turn to speak
                play_user_turn_start_cue()
                
                # Signal program to exit
                exit_program.set()
                response_queue.put(goodbye_message)
                message_queue.task_done()
                continue
            
            # Check if this is a storage command - more robust detection
            if any(cmd in message_lower for cmd in STORE_COMMANDS):
                print("Storage command detected")
                # Set the storage mode flag
                waiting_for_storage.set()
                storage_start_time = time.time()
                # Speak the storage prompt
                speak_storage_prompt()
                # Signal that we're ready for storage input
                response_queue.put("ready_for_storage")
                message_queue.task_done()
                continue
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": message})
            
            # Set the processing flag to prevent new recordings
            is_processing.set()
            
            # Play the user turn end cue to indicate the system is processing
            play_user_turn_end_cue()
            
            # Start streaming response
            print("Starting streaming response...")
            response_text = ""
            for text_chunk in stream_llm_response(conversation_history):
                response_text += text_chunk
            
            # Add assistant response to conversation history
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # Store interaction in memory if available
            if rag_available and rag_handler is not None:
                try:
                    rag_handler.store_interaction(message, response_text)
                except Exception as e:
                    print(f"Error storing interaction: {e}")
            
            # Wait for all pipeline stages to complete
            print("Waiting for text processing to complete...")
            text_chunk_queue.join()
            
            print("Waiting for audio playback to complete...")
            audio_queue.join()
            
            # Add a 4-second delay after the assistant's response and before prompting the user
            print("Waiting 1 second before prompting user...")
            time.sleep(1.0)
            
            # Play the user turn start cue to indicate it's the user's turn to speak
            play_user_turn_start_cue()
            
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
    """Play audio file using the standard audio pipeline"""
    if audio_path and os.path.exists(audio_path):
        try:
            print("Playing audio response...")
            
            # Load audio file
            audio_data, sample_rate = sf.read(audio_path, dtype='float32')
            
            # Normalize if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # If sample rate doesn't match TTS_SAMPLE_RATE, we need to resample
            if sample_rate != TTS_SAMPLE_RATE:
                print(f"Resampling audio from {sample_rate}Hz to {TTS_SAMPLE_RATE}Hz")
                # Simple resampling - for better quality, consider using librosa or scipy
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), int(len(audio_data) * TTS_SAMPLE_RATE / sample_rate)),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Add to audio queue for playback through the standard pipeline
            audio_queue.put(audio_data)
            
            # Wait for audio to finish playing
            while not audio_queue.empty() or audio_playing.is_set():
                time.sleep(0.1)
            
            print("Audio playback complete")
                
        except Exception as e:
            print(f"Error playing audio: {e}")
            traceback.print_exc()

def load_ollama_cache():
    """Load the Ollama response cache from disk"""
    global ollama_cache
    
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    cache_path = get_cache_path("ollama_responses")
    print(f"Looking for cache at: {cache_path}")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                loaded_cache = pickle.load(f)
                
            # Check if the loaded cache is valid
            if not isinstance(loaded_cache, dict):
                print(f"Warning: Loaded cache is not a dictionary, got {type(loaded_cache)}. Starting with empty cache.")
                ollama_cache = {}
                return
                
            # Filter out expired entries
            current_time = time.time()
            valid_entries = {}
            expired_count = 0
            
            for k, v in loaded_cache.items():
                # Check if the entry has the expected format (response, timestamp)
                if not isinstance(v, tuple) or len(v) != 2:
                    print(f"Warning: Cache entry for key {k[:10]}... has invalid format. Skipping.")
                    continue
                    
                # Check if the entry is expired
                if current_time - v[1] < OLLAMA_CACHE_EXPIRY:
                    valid_entries[k] = v
                else:
                    expired_count += 1
            
            ollama_cache = valid_entries
            print(f"Loaded {len(ollama_cache)} valid responses from cache (expired: {expired_count})")
            
            # Print all cache keys for debugging
            if ollama_cache:
                print("All cache keys:")
                for i, k in enumerate(ollama_cache.keys()):
                    print(f"  {i+1}. {k}")
                    
                # Print some sample cache entries for debugging
                print("\nSample cache entries:")
                for i, (k, v) in enumerate(list(ollama_cache.items())[:3]):
                    print(f"  {i+1}. Key: {k}")
                    print(f"     Response: '{v[0][:50]}...'")
                    print(f"     Age: {int(current_time - v[1])}s")
            
            # Test the cache key generation with a few common phrases
            test_phrases = ["hello", "hi there", "hey", "how are you"]
            print("\nTesting cache key generation:")
            for phrase in test_phrases:
                test_messages = [{"role": "user", "content": phrase}]
                test_key = get_ollama_cache_key(LLM_MODEL, test_messages)
                print(f"  Phrase: '{phrase}' -> Key: {test_key}")
                if test_key in ollama_cache:
                    print(f"    ✓ Found in cache!")
                else:
                    print(f"    ✗ Not found in cache")
            
        except Exception as e:
            print(f"Error loading Ollama cache: {e}")
            traceback.print_exc()
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
    global storage_start_time
    
    # Wait for models to be loaded
    print("Waiting for models to load...")
    models_loaded.wait()
    
    p = None
    stream = None
    
    try:
        print("\nInitializing audio recording...")
        p = pyaudio.PyAudio()
        
        # Open the stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print("🎤 Listening... (speak naturally, pause when you're done)")
        
        frames = []
        silent_chunks = 0
        recording = False
        last_activity_time = time.time()
        
        # For debugging - count how many consecutive non-silent chunks we've seen
        consecutive_sound_chunks = 0
        
        print("Ready to record speech!")
        
        while not exit_program.is_set():
            try:
                # Check if stream is valid, restart if needed
                if stream is None:
                    p, stream = restart_audio_stream(p)
                    if stream is None:
                        print("⚠️ Could not restart audio stream, waiting...")
                        time.sleep(1.0)
                        continue
                
                # Check for storage mode timeout
                if waiting_for_storage.is_set() and storage_start_time is not None:
                    elapsed = time.time() - storage_start_time
                    if elapsed > STORAGE_TIMEOUT:
                        print(f"⚠️ Storage mode timed out after {elapsed:.1f} seconds")
                        text_chunk_queue.put("I didn't hear anything to store. Please try again if you want to store something.")
                        waiting_for_storage.clear()
                        is_processing.clear()
                        storage_start_time = None
                        # Reset recording state
                        recording = False
                        frames = []
                        silent_chunks = 0
                
                # If we're in storage mode, print debug info every few seconds
                if waiting_for_storage.is_set() and DEBUG_STORAGE and storage_start_time is not None:
                    if int(time.time()) % 3 == 0:  # Every 3 seconds
                        # Only print once per second (avoid repeated messages)
                        if not hasattr(record_and_transcribe_continuously, 'last_debug_time') or \
                           time.time() - record_and_transcribe_continuously.last_debug_time > 2.5:
                            print(f"👂 Still listening for storage content... (elapsed: {time.time() - storage_start_time:.1f}s)")
                            record_and_transcribe_continuously.last_debug_time = time.time()
                
                # Don't record while processing a message
                if is_processing.is_set():
                    time.sleep(0.1)
                    continue
                
                # Check for audio stream stalling (no activity for 10 seconds)
                if time.time() - last_activity_time > 10.0:
                    print("⚠️ Audio stream may be stalled, restarting...")
                    p, stream = restart_audio_stream(p, stream)
                    last_activity_time = time.time()
                    # Also reset storage mode if it's active
                    if waiting_for_storage.is_set():
                        print("Re-activating storage mode after stream restart")
                        # Clear and then re-set to ensure a fresh state
                        waiting_for_storage.clear()
                        waiting_for_storage.set()
                        storage_start_time = time.time()
                        # Reset recording state
                        recording = False
                        frames = []
                        silent_chunks = 0
                
                # Read audio data
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    last_activity_time = time.time()  # Update activity timestamp
                except Exception as e:
                    print(f"Error reading from audio stream: {e}")
                    time.sleep(0.1)
                    # Try to restart the stream
                    p, stream = restart_audio_stream(p, stream)
                    continue
                
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
                    
                    # Determine the appropriate silence duration based on mode
                    if waiting_for_storage.is_set():
                        # In storage mode, use longer silence duration
                        silence_duration = STORAGE_SILENCE_DURATION
                        if silent_chunks % 10 == 0 and silent_chunks > 0:  # Every ~0.5 seconds
                            print(f"Waiting for storage input: silent for {silent_chunks * CHUNK / RATE:.1f}s...")
                    else:
                        # In normal mode, use standard silence duration
                        silence_duration = SILENCE_DURATION
                    
                    # Calculate the number of chunks that represents the current silence duration
                    max_silent_chunks = int(silence_duration * RATE / CHUNK)
                    
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
                            segments, info = whisper_model.transcribe(
                                audio_file, 
                                beam_size=5,
                                language="en",  # Force English language
                                task="transcribe"  # Explicitly set to transcription task
                            )
                            transcription = " ".join([segment.text for segment in segments])
                            
                            # Filter out hallucinated phrases like "Thanks for watching!"
                            transcription = filter_hallucinations(transcription)
                            
                            if transcription:
                                print(f"You: {transcription}")
                                # Put the transcribed text in the queue for processing
                                message_queue.put(transcription.strip())
                                
                                if waiting_for_storage.is_set():
                                    print("Processing storage input...")
                                    # Reset storage timeout
                                    storage_start_time = None
                                    # Wait for storage to be processed and confirmed
                                    response = response_queue.get()
                                    print(f"Assistant: {response}")
                                    # Make sure recording can continue after storage handling
                                    is_processing.clear()
                                    # Reset for next recording
                                    recording = False
                                    frames = []
                                    silent_chunks = 0
                                    time.sleep(0.2)  # Small pause to ensure clean transition
                                else:
                                    # Check for special signals in the queue without blocking
                                    try:
                                        # Non-blocking check for ready_for_storage signal
                                        response = response_queue.get_nowait()
                                        if response == "ready_for_storage":
                                            # This is just a signal, don't print it
                                            print("Ready for storage input...")
                                            is_processing.clear()
                                            # Reset for next recording
                                            recording = False
                                            frames = []
                                            silent_chunks = 0
                                            continue
                                        else:
                                            # If it's a real response, print it
                                            print(f"Assistant: {response}")
                                    except queue.Empty:
                                        # If no signal is waiting, proceed to normal processing
                                        print("Processing your message...")
                                        response = response_queue.get()  # This will block until a response is available
                                        print(f"Assistant: {response}")
                                    
                                    # The processing flag should be cleared after we get the response
                                    is_processing.clear()
                                
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

def play_user_turn_start_cue():
    """Play an audio cue to indicate it's the user's turn to speak"""
    try:
        if os.path.exists(USER_TURN_START_SOUND):
            print("Playing user turn start cue...")
            # Load audio file
            audio_data, sample_rate = sf.read(USER_TURN_START_SOUND, dtype='float32')
            
            # Normalize if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Resample if needed to match TTS sample rate
            if sample_rate != TTS_SAMPLE_RATE:
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), int(len(audio_data) * TTS_SAMPLE_RATE / sample_rate)),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Add to audio queue for playback through the standard pipeline
            audio_queue.put(audio_data)
            
            # Only wait briefly to ensure audio starts playing
            # This allows the function to return while audio is still playing
            time.sleep(0.1)
    except Exception as e:
        print(f"Error playing user turn start cue: {e}")
        traceback.print_exc()

def play_user_turn_end_cue():
    """Play an audio cue to indicate the user's turn has ended"""
    try:
        if os.path.exists(USER_TURN_END_SOUND):
            print("Playing user turn end cue...")
            # Load audio file
            audio_data, sample_rate = sf.read(USER_TURN_END_SOUND, dtype='float32')
            
            # Normalize if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Resample if needed to match TTS sample rate
            if sample_rate != TTS_SAMPLE_RATE:
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), int(len(audio_data) * TTS_SAMPLE_RATE / sample_rate)),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Add to audio queue for playback through the standard pipeline
            audio_queue.put(audio_data)
            
            # Only wait briefly to ensure audio starts playing
            # This allows the function to return while audio is still playing
            time.sleep(0.1)
    except Exception as e:
        print(f"Error playing user turn end cue: {e}")
        traceback.print_exc()

def filter_hallucinations(transcription):
    """Filter out common hallucinated phrases from the transcription"""
    # List of known hallucinated phrases to filter out
    hallucination_phrases = [
        "Thanks for watching!",
        "Thanks for watching.",
        "Thank you for watching!",
        "Thank you for watching.",
        "Don't forget to subscribe!",
        "Don't forget to like and subscribe!",
        "Please like and subscribe!",
        "Like and subscribe!",
        "Subscribe to our channel!",
        "Hit the like button!",
        "Leave a comment below!",
        "Check the description below!",
        "Follow us on social media!",
        "See you in the next video!",
        "Until next time!",
        "Bon Appetit!"
    ]
    
    # Check if the transcription consists entirely of hallucinated phrases
    transcription_lower = transcription.lower().strip()
    for phrase in hallucination_phrases:
        if transcription_lower == phrase.lower():
            print(f"Filtered out hallucinated phrase: '{transcription}'")
            return ""
    
    # If the hallucinated phrase is part of a longer transcription, remove it
    filtered_transcription = transcription
    for phrase in hallucination_phrases:
        filtered_transcription = filtered_transcription.replace(phrase, "")
    
    # Clean up any double spaces created by removing phrases
    filtered_transcription = " ".join(filtered_transcription.split())
    
    # If we made changes, log it
    if filtered_transcription != transcription:
        print(f"Filtered hallucination from: '{transcription}' to '{filtered_transcription}'")
    
    return filtered_transcription

def save_ollama_cache():
    """Save the Ollama response cache to disk"""
    global ollama_cache
    
    if not ollama_cache:
        print("Cache is empty, nothing to save")
        return
        
    try:
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        cache_path = get_cache_path("ollama_responses")
        with open(cache_path, 'wb') as f:
            pickle.dump(ollama_cache, f)
        print(f"Saved {len(ollama_cache)} responses to cache at {cache_path}")
        return True
    except Exception as e:
        print(f"Error saving Ollama cache: {e}")
        traceback.print_exc()
        return False

def seed_cache_with_common_phrases():
    """Seed the cache with common greetings and phrases to ensure they're available"""
    global ollama_cache
    
    # Only seed if the cache is empty or has very few entries
    if len(ollama_cache) > 10:
        print("Cache already has sufficient entries, skipping seeding")
        return
        
    print("Seeding cache with common phrases...")
    
    # Common greetings and their responses
    common_phrases = {
        "hello": "Hello! How can I help you today?",
        "hi": "Hi there! How can I assist you?",
        "hey": "Hey! What can I do for you today?",
        "how are you": "I'm doing well, thank you for asking! How can I help you?",
        "good morning": "Good morning! How can I assist you today?",
        "good afternoon": "Good afternoon! What can I help you with?",
        "good evening": "Good evening! How may I assist you?",
        "thanks": "You're welcome! Is there anything else I can help with?",
        "thank you": "You're welcome! Let me know if you need anything else.",
        "bye": "Goodbye! Have a great day!",
        "goodbye": "Goodbye! It was nice chatting with you."
    }
    
    # Current time for timestamp
    current_time = time.time()
    
    # Add each phrase to the cache
    for phrase, response in common_phrases.items():
        # Generate a cache key for this phrase
        test_messages = [{"role": "user", "content": phrase}]
        cache_key = get_ollama_cache_key(LLM_MODEL, test_messages)
        
        # Only add if not already in cache
        if cache_key not in ollama_cache:
            ollama_cache[cache_key] = (response, current_time)
            print(f"Added '{phrase}' to cache with key: {cache_key}")
    
    # Save the seeded cache
    save_ollama_cache()
    print(f"Cache seeded with {len(common_phrases)} common phrases")

def main():
    """Main function to run the continuous speech conversation"""
    try:
        # Load Ollama cache
        load_ollama_cache()
        
        # Seed the cache with common phrases
        seed_cache_with_common_phrases()
        
        # Load models
        print("Loading models...")
        load_models()
        
        # Start the message processing thread
        # This will also start the audio generation and playback threads
        processing_thread = threading.Thread(target=process_messages)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Give the threads a moment to initialize
        time.sleep(1.0)
        
        # Say welcome message
        say_welcome_message()
        
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
                save_ollama_cache()
        except Exception as e:
            print(f"Error saving cache on exit: {e}")

if __name__ == "__main__":
    print("Starting continuous speech conversation...")
    print("Press Ctrl+C to exit")
    main()
