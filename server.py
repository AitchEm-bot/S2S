from flask import Flask, request, render_template, Response, stream_with_context, jsonify, send_from_directory
import os
from handling import handlers
from flask_cors import CORS
from ollama import OllamaChat
import json
import requests
import traceback
from entity_tracker import EntityTracker
import re
import time
import uuid
import hashlib
import pickle

save_text_to_file = handlers.save_text_to_file
transcribe_audio = handlers.transcribe_audio
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

integer_list = []

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache settings
CACHE_DIR = "model_cache"  # Directory to store model cache
OLLAMA_CACHE_EXPIRY = 3600  # Cache Ollama responses for 1 hour (in seconds)
ollama_cache = {}  # In-memory cache for Ollama responses
DEBUG_MODE = False  # Set to True to enable detailed cache debugging

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the chat client
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Main conversation model
ollama_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1")
# Smaller model for summarization tasks
summarization_model = os.environ.get("SUMMARIZATION_MODEL", "mistral")

# System prompt for the assistant
system_prompt = """You are a helpful assistant that answers questions and helps with tasks. 
Your responses should:
1. Be based only on the provided context and your training
2. Clearly indicate when you're uncertain about information
3. Avoid making up facts or speculating without evidence
4. Request clarification when the query is ambiguous

Do not say anything like "it seems you mentioned that..." or anything of the sort.
Your role is to be a good listener and a problem solver for the user's emotions while maintaining factual accuracy."""

# Options for the Ollama model
model_options = {
    "temperature": 0.3,
    "top_p": 0.7,
    "num_ctx": 8192  # Increased context window for deepseek-r1
}

# RAG configuration
rag_config = {
    "storage_min": 0.3,           # Minimum score to store anything
    "ephemeral_max": 0.4,         # Maximum score for ephemeral memory
    "short_term_max": 0.7,        # Maximum score for short-term memory
    "retrieval_min": 0.0,         # Set to 0 to allow all queries to be used for retrieval
    "similarity_max": 0.8,        # Maximum similarity to consider duplicate
    "ephemeral_similarity": 0.5,  # Minimum similarity for ephemeral retrieval
    "key_points_min_length": 30,  # Minimum length to trigger key points extraction
    "short_term_expiry_days": 7   # Days before short-term memory expires
}

# Initialize the chat client with the configuration
# Check if we're in the main Flask process or the reloader
is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'

# Only initialize RAG fully in the main process to avoid duplicate maintenance
if is_reloader:
    print("Main Flask process: Initializing RAG with maintenance")
    ollama_chat = OllamaChat(
        base_url=ollama_base_url,
        model=ollama_model,
        system_prompt=system_prompt,
        options=model_options,
        rag_config=rag_config
    )
else:
    print("Reloader process: Initializing RAG without maintenance")
    # Create a modified config that disables maintenance
    no_maintenance_config = rag_config.copy()
    no_maintenance_config["skip_maintenance"] = True
    
    ollama_chat = OllamaChat(
        base_url=ollama_base_url,
        model=ollama_model,
        system_prompt=system_prompt,
        options=model_options,
        rag_config=no_maintenance_config
    )

# Load cache on startup
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

def get_ollama_cache_key(model, messages, max_history=1):
    """Generate a cache key for Ollama requests
    
    Args:
        model (str): The model name
        messages (list): The full message history
        max_history (int): Maximum number of previous messages to include in the cache key
                          (1 means just the current message, 2 means current + previous, etc.)
    """
    # For caching, we only want to consider the last few messages
    # This prevents the cache from being too specific and allows for reuse
    if max_history > 0 and len(messages) > 0:
        # Always include the system message if present
        system_message = None
        if messages and messages[0]['role'] == 'system':
            system_message = messages[0]
        
        # Get the last N messages (where N is max_history)
        recent_messages = messages[-max_history:]
        
        # Add the system message back if it was present
        if system_message:
            recent_messages = [system_message] + recent_messages
        
        # Use these messages for the cache key
        cache_messages = recent_messages
    else:
        # Use all messages if max_history is 0 or negative
        cache_messages = messages
    
    # Create a string representation of the request
    request_str = f"{model}_{json.dumps(cache_messages, sort_keys=True)}"
    
    # Hash it to create a cache key
    hash_obj = hashlib.md5(request_str.encode())
    return hash_obj.hexdigest()

# Load cache on startup
load_ollama_cache()

# Interaction counter for unique IDs
interaction_counter = 0

def new_int_name():
    for name in os.listdir("uploads"):
        try:
            print(name[:-4])
            # int(name[:-4])
            integer_list.append(int(name[:-4]))
        except:
            pass
    return max(integer_list)
            

@app.route("/")
def index():
    return render_template("record.html")

@app.route("/home")
def load_home_page():
    return render_template("record.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files from the templates directory"""
    return send_from_directory("templates", filename)

@app.route("/listen_audio", methods=["POST"])
def listen_audio():
    if "audio" not in request.files:
        return "No audio file found", 400
    
    audio_file = request.files["audio"]
    filename = audio_file.filename.strip() if audio_file.filename.strip() else "audio.wav"
    audio_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(audio_path)
    
    # Get the original transcription
    original_transcription = transcribe_audio(f"uploads/{filename}")
    
    # Check if transcription is empty
    if not original_transcription.strip():
        return {
            "transcription": original_transcription,
            "ollama_response": "Transcription is empty, not stored.",
            "debug_info": {
                "stored_in_rag": False,
                "reason": "Empty transcription"
            }
        }
    
    # Save the original transcription to a file
    save_text_to_file(f"transcriptions/{filename[:-4]}.txt", original_transcription)
    
    # Process the transcription to extract key points
    try:
        # Ask Ollama to summarize or extract key points
        summarize_prompt = "Extract the most important points from this transcription in bullet points:\n\n" + original_transcription + "\n\nRespond ONLY with bullet points of key information. No introductions or explanations needed."
        
        # Use a non-streaming request to get the processed version
        messages = [
            {"role": "system", "content": "Extract only the essential information from text. Respond with concise bullet points only."},
            {"role": "user", "content": summarize_prompt}
        ]
        
        # Get the processed transcription from Ollama
        response = requests.post(
            f"{ollama_base_url}/api/chat",
            json={
                "model": summarization_model,  # Use the smaller model for summarization
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for more focused summaries
                    "top_p": 0.9
                }
            }
        )
        
        if response.status_code == 200:
            processed_transcription = response.json()["message"]["content"]
            print(f"Processed transcription: {processed_transcription}")
        else:
            print(f"Error from Ollama API: {response.status_code}")
            processed_transcription = original_transcription
    except Exception as e:
        print(f"Error processing transcription: {str(e)}")
        # Fall back to original transcription if processing fails
        processed_transcription = original_transcription
        
    # Store the original transcription for importance evaluation
    stored = ollama_chat.rag.store_message(
        original_transcription, 
        "user",
        0.7  # Use a fixed importance score for transcriptions
    )
    
    if stored:
        # Add original transcription to immediate context
        ollama_chat.context.append({
            "role": "user", 
            "content": original_transcription, 
            "source": "transcription"
        })
        print(f"Added to immediate context. Context length: {len(ollama_chat.context)}")
        
        return {
            "transcription": original_transcription,
            "processed_transcription": processed_transcription,
            "ollama_response": "Transcription processed and stored in memory.",
            "debug_info": {
                "stored_in_rag": True,
                "context_length": len(ollama_chat.context)
            }
        }
    else:
        return {
            "transcription": original_transcription,
            "ollama_response": "Transcription received but not stored due to low importance.",
            "debug_info": {
                "reason": "Failed storage criteria"
            }
        }

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
            
        # Calculate importance score for logging purposes
        importance = ollama_chat.rag.importance_score(message, role="user")
        print(f"Message importance: {importance:.2f}")
        
        # Extract entities from the message
        entities = ollama_chat.rag.entity_tracker.extract_entities(message)
        if entities:
            print(f"Extracted entities: {len(entities)}")
            for entity in entities[:5]:  # Show top 5 entities
                print(f"- {entity['text']} ({entity['type']})")
        
        # Analyze if this is a question or information-seeking query
        query_analysis = ollama_chat.rag.analyze_query(message)
        
        # Determine if we should retrieve context
        should_retrieve = query_analysis['retrieval_recommended'] or importance > ollama_chat.rag.thresholds['retrieval_min']
        
        # Get relevant context if needed
        context = ""
        if should_retrieve:
            context = ollama_chat.rag.search_relevant_context(message, force_retrieval=False)
            if context:
                print(f"Retrieved context: {len(context)} characters")
            else:
                print("No relevant context found")
        
        # Get entity summaries if available and relevant
        entity_context = ""
        if query_analysis['retrieval_recommended']:
            query_entities = ollama_chat.rag.entity_tracker.find_entities_in_query(message)
            if query_entities:
                for entity in query_entities:
                    summary = ollama_chat.rag.entity_tracker.get_entity_summary(entity['text'])
                    if summary:
                        entity_context += f"\nEntity: {entity['text']} - {summary}\n"
        
        # Combine contexts
        if entity_context:
            if context:
                context += "\n\nEntity Information:\n" + entity_context
            else:
                context = "Entity Information:\n" + entity_context
        
        # Prepare messages for cache key generation
        messages = ollama_chat.context.copy()
        messages.append({"role": "user", "content": message})
        
        # Check cache before streaming
        # Use only the current message for caching to avoid context-specific responses
        cache_key = get_ollama_cache_key(ollama_chat.model, messages, max_history=1)
        current_time = time.time()
        
        # Debug logging for cache (only visible in server logs)
        if DEBUG_MODE:
            print(f"Cache key: {cache_key}")
            print(f"Cache hit: {cache_key in ollama_cache}")
            if cache_key in ollama_cache:
                cached_response, timestamp = ollama_cache[cache_key]
                print(f"Cache age: {int(current_time - timestamp)} seconds")
                print(f"Cache valid: {current_time - timestamp < OLLAMA_CACHE_EXPIRY}")
                print(f"Cache response (first 50 chars): {cached_response[:50]}...")
        else:
            # Minimal logging in non-debug mode
            if cache_key in ollama_cache:
                print("Using optimized response generation")
        
        # Stream the response
        def generate():
            first_chunk = True
            in_think_block = False
            buffer = ""
            
            # Store the message in memory if important enough
            if importance > ollama_chat.rag.thresholds['storage_min']:
                ollama_chat.rag.store_message(message, "user", importance)
            
            # Check if we have a cached response
            if cache_key in ollama_cache:
                cached_response, timestamp = ollama_cache[cache_key]
                # Check if the cache is still valid and not empty
                if current_time - timestamp < OLLAMA_CACHE_EXPIRY and cached_response and cached_response.strip():
                    if DEBUG_MODE:
                        print(f"Using cached response (cached {int(current_time - timestamp)} seconds ago)")
                    else:
                        print("Generating response...")
                    
                    # Add the cached response to the context
                    ollama_chat.add_to_context("assistant", cached_response)
                    
                    # Stream the cached response in chunks to simulate typing effect
                    # This ensures the frontend experience is consistent with non-cached responses
                    
                    # First, send a special flag to indicate this is a cached response
                    # This is only for internal processing and won't be visible to the user
                    yield f"data: {json.dumps({'cached_start': True})}\n\n"
                    
                    # Split the response into characters to simulate typing
                    # We'll send chunks of characters to simulate a natural typing speed
                    chunk_size = 5  # Characters per chunk
                    
                    # Process the response character by character
                    for i in range(0, len(cached_response), chunk_size):
                        # Get the next chunk of characters
                        chunk = cached_response[i:i+chunk_size]
                        # Send the chunk without the 'cached' flag
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                        # Add a small delay to simulate typing speed
                        time.sleep(0.02)  # 20ms delay between chunks
                    
                    # Finally, send a flag to indicate the end of the cached response
                    # This is only for internal processing and won't be visible to the user
                    yield f"data: {json.dumps({'cached_end': True})}\n\n"
                    
                    return
            
            # If no cache hit, stream from Ollama
            for chunk in ollama_chat.chat(message, stream=True, context=context):
                # Check for <think> tags and filter content between them
                if "<think>" in chunk:
                    # Split the chunk at <think>
                    parts = chunk.split("<think>", 1)
                    # Send the part before <think>
                    if parts[0]:
                        # Always trim leading whitespace from the first chunk
                        if first_chunk:
                            parts[0] = parts[0].lstrip()
                            first_chunk = False
                        yield f"data: {json.dumps({'chunk': parts[0]})}\n\n"
                    # Mark that we're in a think block
                    in_think_block = True
                    # Save any content after <think> in case it contains </think>
                    buffer = parts[1] if len(parts) > 1 else ""
                elif "</think>" in chunk and in_think_block:
                    # Split the chunk at </think>
                    parts = chunk.split("</think>", 1)
                    # Ignore the part before </think>
                    # Mark that we're no longer in a think block
                    in_think_block = False
                    # Send the part after </think>, trimming leading whitespace
                    if len(parts) > 1 and parts[1]:
                        # Always trim leading whitespace after a think block
                        trimmed_part = parts[1].lstrip()
                        if trimmed_part:
                            yield f"data: {json.dumps({'chunk': trimmed_part})}\n\n"
                    buffer = ""
                elif in_think_block:
                    # We're in a think block, so don't send anything
                    buffer += chunk
                    # Check if the buffer now contains </think>
                    if "</think>" in buffer:
                        parts = buffer.split("</think>", 1)
                        in_think_block = False
                        if parts[1]:  # Send content after </think>, trimming leading whitespace
                            trimmed_part = parts[1].lstrip()
                            if trimmed_part:
                                yield f"data: {json.dumps({'chunk': trimmed_part})}\n\n"
                        buffer = ""
                else:
                    # Normal chunk, not in a think block
                    # Always trim leading whitespace from the first chunk
                    if first_chunk:
                        chunk = chunk.lstrip()
                        first_chunk = False
                        
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Store the complete response in cache
            complete_response = ollama_chat.get_last_response()
            
            # Only cache non-empty responses
            if complete_response and complete_response.strip():
                print(f"Caching response: '{complete_response[:100]}...'")
                ollama_cache[cache_key] = (complete_response, current_time)
                
                # Save the cache periodically
                if len(ollama_cache) % 5 == 0:
                    try:
                        cache_path = get_cache_path("ollama_responses")
                        with open(cache_path, 'wb') as f:
                            pickle.dump(ollama_cache, f)
                        print(f"Saved {len(ollama_cache)} responses to cache")
                    except Exception as e:
                        print(f"Error saving cache: {e}")
            else:
                print("Response is empty, not caching")
            
        return Response(generate(), mimetype='text/event-stream')
    except Exception as e:
        error_msg = f"Error in chat: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

@app.route("/reset_context", methods=["POST"])
def reset_context():
    try:
        # Handle the cache file
        cache_path = get_cache_path("ollama_responses")
        
        # Delete the cache file if it exists
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"Deleted cache file: {cache_path}")
            except Exception as e:
                print(f"Error deleting cache file: {e}")
        
        # Clear the in-memory cache
        ollama_cache.clear()
        print("In-memory cache cleared")
        
        # Reset the conversation context
        ollama_chat.reset_context()
        
        # Clear the RAG memory
        rag_result = ollama_chat.rag.clear_collection()
        
        # Clear the entity tracker
        ollama_chat.rag.entity_tracker = EntityTracker()
        
        # Get data from request, handling both JSON and non-JSON requests
        try:
            data = request.json or {}
        except:
            # If request.json fails (not JSON content), use an empty dict
            data = {}
        
        # Clear collections if requested (default to True if not specified)
        if data.get('clear_memory', True):
            message = "Both conversation context and long-term memory have been cleared successfully. Response cache has been permanently deleted."
        else:
            message = "Conversation context has been cleared, but long-term memory is preserved. Response cache has been permanently deleted."
            
        return jsonify({'status': 'success', 'message': message})
    except Exception as e:
        error_msg = f"Error in reset_context: {e}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"error": f"Could not reset context. {str(e)}"}), 500

@app.route("/get_chat_history")
def get_chat_history():
    """Return the chat history"""
    return jsonify({
        "history": ollama_chat.context
    })

@app.route("/check_rag")
def check_rag():
    """Endpoint to check RAG contents"""
    try:
        print("Checking RAG contents...")
        
        # Get data from all memory types
        long_term_results = ollama_chat.rag.long_term_memory.get()
        short_term_results = ollama_chat.rag.short_term_memory.get()
        ephemeral_memory = ollama_chat.rag.ephemeral_memory
        
        # Process long-term memory
        long_term_messages = []
        if 'metadatas' in long_term_results and 'documents' in long_term_results and long_term_results['metadatas']:
            for i, (metadata, document) in enumerate(zip(long_term_results['metadatas'], long_term_results['documents'])):
                long_term_messages.append(f"{i+1}. {document}")
        
        # Process short-term memory
        short_term_messages = []
        if 'metadatas' in short_term_results and 'documents' in short_term_results and short_term_results['metadatas']:
            for i, (metadata, document) in enumerate(zip(short_term_results['metadatas'], short_term_results['documents'])):
                short_term_messages.append(f"{i+1}. {document}")
        
        # Process ephemeral memory
        ephemeral_messages = []
        for i, item in enumerate(ephemeral_memory):
            ephemeral_messages.append(f"{i+1}. {item['text']}")
        
        response_data = {
            "message": "RAG Contents By Memory Type",
            "long_term": {
                "count": len(long_term_messages),
                "messages": long_term_messages
            },
            "short_term": {
                "count": len(short_term_messages),
                "messages": short_term_messages
            },
            "ephemeral": {
                "count": len(ephemeral_messages),
                "messages": ephemeral_messages
            },
            "buffer_size": len(ollama_chat.rag.storage_buffer)
        }
        
        print(f"RAG contents summary: {len(long_term_messages)} long-term, {len(short_term_messages)} short-term, {len(ephemeral_messages)} ephemeral")
        return jsonify(response_data)
    except Exception as e:
        error_msg = f"Error in check_rag: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg})

@app.route("/rag_stats", methods=['GET'])
def rag_stats():
    try:
        # Get stats from different memory types
        long_term_count = len(ollama_chat.rag.long_term_memory.get()['ids']) if ollama_chat.rag.long_term_memory.get() else 0
        short_term_count = len(ollama_chat.rag.short_term_memory.get()['ids']) if ollama_chat.rag.short_term_memory.get() else 0
        ephemeral_count = len(ollama_chat.rag.ephemeral_memory)
        buffer_size = len(ollama_chat.rag.storage_buffer)
        
        # Get all tags
        all_tags = ollama_chat.rag.get_all_tags()
        
        return jsonify({
            'long_term_count': long_term_count,
            'short_term_count': short_term_count,
            'ephemeral_count': ephemeral_count,
            'buffer_size': buffer_size,
            'all_tags': all_tags
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories/tags', methods=['GET'])
def get_all_tags():
    try:
        tags = ollama_chat.rag.get_all_tags()
        return jsonify({'tags': tags})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories/by_tag/<tag>', methods=['GET'])
def get_memories_by_tag(tag):
    try:
        max_results = request.args.get('max_results', default=10, type=int)
        memories = ollama_chat.rag.get_memories_by_tag(tag, max_results)
        
        # Format the response
        formatted_memories = []
        for memory in memories:
            formatted_memories.append({
                'id': memory['id'],
                'text': memory['text'],
                'source': memory['source'],
                'timestamp': memory['metadata'].get('timestamp', 0),
                'tags': memory['metadata'].get('tags', []),
                'type': memory['metadata'].get('type', 'unknown')
            })
        
        return jsonify({
            'tag': tag,
            'count': len(formatted_memories),
            'memories': formatted_memories
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories/feedback', methods=['POST'])
def provide_feedback():
    try:
        data = request.json
        memory_id = data.get('memory_id')
        feedback_type = data.get('feedback_type')
        memory_source = data.get('memory_source', 'long_term')
        
        if not memory_id or not feedback_type:
            return jsonify({'error': 'Missing required parameters'}), 400
            
        if feedback_type not in ['useful', 'not_useful', 'incorrect', 'outdated']:
            return jsonify({'error': 'Invalid feedback type'}), 400
            
        success = ollama_chat.rag.process_user_feedback(memory_id, feedback_type, memory_source)
        
        if success:
            return jsonify({'status': 'success', 'message': f'Feedback ({feedback_type}) processed for memory {memory_id}'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to process feedback'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.teardown_appcontext
def save_cache_on_shutdown(exception=None):
    """Save the cache when the application context tears down (server shutdown)"""
    if ollama_cache:
        try:
            cache_path = get_cache_path("ollama_responses")
            with open(cache_path, 'wb') as f:
                pickle.dump(ollama_cache, f)
            print(f"Saved {len(ollama_cache)} responses to cache on shutdown")
        except Exception as e:
            print(f"Error saving cache on shutdown: {e}")

@app.route("/debug/context_and_cache", methods=["GET"])
def debug_context_and_cache():
    """Debug endpoint to check the current state of the context window and cache"""
    try:
        # Get the current context
        context = ollama_chat.context
        
        # Get cache file info
        cache_path = get_cache_path("ollama_responses")
        cache_file_exists = os.path.exists(cache_path)
        cache_file_size = os.path.getsize(cache_path) if cache_file_exists else 0
        
        # Get cache stats
        cache_stats = {
            "total_entries": len(ollama_cache),
            "cache_size_kb": sum(len(pickle.dumps(item)) for item in ollama_cache.values()) / 1024,
            "oldest_entry": min([timestamp for _, (_, timestamp) in ollama_cache.items()]) if ollama_cache else None,
            "newest_entry": max([timestamp for _, (_, timestamp) in ollama_cache.items()]) if ollama_cache else None,
            "cache_file": {
                "exists": cache_file_exists,
                "path": cache_path,
                "size_kb": cache_file_size / 1024 if cache_file_exists else 0
            }
        }
        
        if cache_stats["oldest_entry"]:
            cache_stats["oldest_entry_age"] = time.time() - cache_stats["oldest_entry"]
            cache_stats["oldest_entry"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_stats["oldest_entry"]))
            
        if cache_stats["newest_entry"]:
            cache_stats["newest_entry_age"] = time.time() - cache_stats["newest_entry"]
            cache_stats["newest_entry"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_stats["newest_entry"]))
        
        # Get sample cache entries (up to 5)
        sample_entries = []
        for key, (response, timestamp) in list(ollama_cache.items())[:5]:
            # Try to reconstruct what message generated this cache key
            # This is just a best-effort attempt and may not be accurate
            sample_entries.append({
                "key": key,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
                "age_seconds": int(time.time() - timestamp),
                "response_preview": response[:100] + "..." if len(response) > 100 else response,
                "response_length": len(response)
            })
        
        cache_stats["sample_entries"] = sample_entries
        
        # Get RAG stats
        rag_stats = {
            "entity_count": len(ollama_chat.rag.entity_tracker.entities) if hasattr(ollama_chat.rag, 'entity_tracker') else 0,
        }
        
        return jsonify({
            "context_window": {
                "length": len(context),
                "messages": context,
            },
            "cache": cache_stats,
            "rag": rag_stats
        })
    except Exception as e:
        error_msg = f"Error getting debug info: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

@app.route("/debug/clear_cache", methods=["POST"])
def clear_cache():
    """Manually clear the cache"""
    try:
        # Handle the cache file
        cache_path = get_cache_path("ollama_responses")
        
        # Delete the cache file if it exists
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"Deleted cache file: {cache_path}")
            except Exception as e:
                print(f"Error deleting cache file: {e}")
                return jsonify({"status": "error", "message": f"Error deleting cache file: {str(e)}"}), 500
        
        # Clear the in-memory cache
        ollama_cache.clear()
        print("In-memory cache cleared")
        
        return jsonify({
            "status": "success", 
            "message": "Cache cleared successfully",
            "details": {
                "cache_file_deleted": not os.path.exists(cache_path),
                "in_memory_cache_cleared": True,
                "cache_file_path": cache_path
            }
        })
    except Exception as e:
        error_msg = f"Error clearing cache: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"status": "error", "message": error_msg}), 500

if __name__ == "__main__":
    app.run(debug=True, port=9999, host="0.0.0.0")
