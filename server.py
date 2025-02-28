from flask import Flask, request, render_template, Response, stream_with_context, jsonify, send_from_directory
import os
from handling import handlers
from flask_cors import CORS
from ollama import OllamaChat
import json
import requests

save_text_to_file = handlers.save_text_to_file
transcribe_audio = handlers.transcribe_audio
app = Flask(__name__)
CORS(app)

integer_list = []

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the chat client
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# ollama_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1")
ollama_model = os.environ.get("OLLAMA_MODEL", "Mistral")

# System prompt for the assistant
system_prompt = """You are a therapist that will listen to the user and help them with
their problems. Answer questions concisely and accurately.
If you don't know something, say so rather than making up information.
For code examples, provide clear explanations and well-commented code."""

# Options for the Ollama model
model_options = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_ctx": 8192  # Increased context window for deepseek-r1
}

# RAG configuration
rag_config = {
    "storage_min": 0.3,           # Minimum score to store anything
    "ephemeral_max": 0.4,         # Maximum score for ephemeral memory
    "short_term_max": 0.7,        # Maximum score for short-term memory
    "retrieval_min": 0.3,         # Minimum query importance for retrieval
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
    
    transcription = transcribe_audio(f"uploads/{filename}")
    
    # Check if transcription is empty
    if not transcription.strip():
        return {
            "transcription": transcription,
            "ollama_response": "Transcription is empty, not stored.",
            "debug_info": {
                "stored_in_rag": False,
                "reason": "Empty transcription"
            }
        }
    
    save_text_to_file(f"transcriptions/{filename[:-4]}.txt", transcription)
    
    # Store summarized version in RAG - explicitly set source to "audio"
    stored = ollama_chat.rag.store_transcription(transcription, source="audio")
    
    if stored:
        # Add original transcription to immediate context
        ollama_chat.context.append({
            "role": "user", 
            "content": transcription, 
            "source": "transcription"
        })
        print(f"2. Added to immediate context. Context length: {len(ollama_chat.context)}")
        
        return {
            "transcription": transcription,
            "ollama_response": "Transcription processed and summarized for long-term memory.",
            "debug_info": {
                "stored_in_rag": True,
                "context_length": len(ollama_chat.context)
            }
        }
    else:
        return {
            "transcription": transcription,
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
            
        # Stream the response
        def generate():
            first_chunk = True
            in_think_block = False
            buffer = ""
            
            for chunk in ollama_chat.chat(message, stream=True):
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
                
        return Response(generate(), mimetype='text/event-stream')
    except Exception as e:
        error_msg = f"Error in chat: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg})

@app.route("/reset_context", methods=["POST"])
def reset_context():
    """Reset both immediate context and RAG memory"""
    response = ollama_chat.reset_context()
    return {"message": response}

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
        if 'metadatas' in long_term_results and long_term_results['metadatas']:
            for i, metadata in enumerate(long_term_results['metadatas']):
                long_term_messages.append(f"{i+1}. {metadata['text']}")
        
        # Process short-term memory
        short_term_messages = []
        if 'metadatas' in short_term_results and short_term_results['metadatas']:
            for i, metadata in enumerate(short_term_results['metadatas']):
                short_term_messages.append(f"{i+1}. {metadata['text']}")
        
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

if __name__ == "__main__":
    app.run(debug=True, port=9999, host="0.0.0.0")
