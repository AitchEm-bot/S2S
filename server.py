from flask import Flask, request, render_template, Response, stream_with_context, jsonify
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

# Initialize Ollama chat with RAG
ollama_chat = OllamaChat()

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

@app.route("/listen_audio", methods=["POST"])
def listen_audio():
    if "audio" not in request.files:
        return "No audio file found", 400
    
    audio_file = request.files["audio"]
    filename = audio_file.filename.strip() if audio_file.filename.strip() and audio_file.filename != ".wav" else str(new_int_name()+1)+".wav"
    audio_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(audio_path)
    
    transcription = transcribe_audio(f"uploads/{filename}")
    save_text_to_file(f"transcriptions/{filename[:-4]}.txt", transcription)
    
    # Add transcription to context and store in RAG
    ollama_chat.context.append({"role": "user", "content": transcription, "source": "transcription"})
    
    return {"transcription": transcription, "ollama_response": "Transcription added to context. You can now chat about it."}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    if not message:
        return {"error": "No message provided"}, 400
    
    print(f"Received message: {message}")
    
    def generate():
        try:
            # Use ollama_chat.chat() instead of direct API call
            response = ollama_chat.chat(message)
            
            if response:
                # Stream the response back
                yield json.dumps({"response": response}) + '\n'
            else:
                yield json.dumps({"error": "No response received"}) + '\n'
                
        except Exception as e:
            print(f"Error in chat endpoint: {str(e)}")
            yield json.dumps({"error": str(e)}) + '\n'

    return Response(stream_with_context(generate()), mimetype='application/json')

@app.route("/reset_context", methods=["POST"])
def reset_context():
    """Reset conversation context while preserving RAG memory"""
    print("Attempting to reset context and RAG...")  # Debug print
    response = ollama_chat.reset_context()
    print(f"Reset response: {response}")  # Debug print
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
        print("Checking RAG contents...")  # Debug print
        all_results = ollama_chat.rag.collection.get()
        
        stored_messages = []
        if all_results and 'metadatas' in all_results:
            for i, metadata in enumerate(all_results['metadatas']):
                stored_messages.append(f"{i+1}. User: {metadata['text']}")
        
        response_data = {
            "message": "RAG Contents (User Messages)",
            "count": len(stored_messages),
            "messages": stored_messages
        }
        print(f"RAG contents: {response_data}")  # Debug print
        return jsonify(response_data)
    except Exception as e:
        error_msg = f"Error in check_rag: {str(e)}"
        print(error_msg)  # Debug print
        return jsonify({"error": error_msg})

if __name__ == "__main__":
    app.run(debug=True, port=9999, host="0.0.0.0")
