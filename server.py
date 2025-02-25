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

# Initialize Ollama chat
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
    
    # Add transcription to Ollama's context without generating an immediate response
    ollama_chat.add_to_context(transcription)
    
    return {"transcription": transcription, "ollama_response": "Transcription added to context. You can now chat about it."}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    if not message:
        return {"error": "No message provided"}, 400
    
    print(f"Received message: {message}")  # Debug log
    
    def generate():
        try:
            response = requests.post(
                f"http://localhost:11434/api/chat",
                json={
                    "model": ollama_chat.model_name,
                    "messages": ollama_chat.context + [{"role": "user", "content": message}],
                    "stream": True
                },
                stream=True
            )
            
            current_message = ""
            first_chunk = True  # Flag to track the first chunk
            
            for line in response.iter_lines():
                if line:
                    try:
                        response_json = json.loads(line.decode('utf-8'))
                        if "message" in response_json:
                            chunk = response_json["message"]["content"]
                            
                            # Remove leading space only from the first chunk
                            if first_chunk and chunk:
                                chunk = chunk.lstrip(' ')
                                first_chunk = False
                            
                            current_message += chunk
                            # Stream each chunk to the frontend
                            yield json.dumps({"response": chunk}) + '\n'
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {str(e)}, line: {line}")
            
            # After streaming is complete, add to context
            if current_message:
                ollama_chat.context.append({"role": "assistant", "content": current_message})
                
        except Exception as e:
            print(f"Error in chat endpoint: {str(e)}")  # Debug log
            yield json.dumps({"error": str(e)}) + '\n'

    return Response(stream_with_context(generate()), mimetype='application/json')

@app.route("/reset_context", methods=["POST"])
def reset_context():
    response = ollama_chat.reset_context()
    return {"message": response}

@app.route("/get_chat_history")
def get_chat_history():
    """Return the chat history"""
    return jsonify({
        "history": ollama_chat.context
    })

if __name__ == "__main__":
    app.run(debug=True, port=9999, host="0.0.0.0")
