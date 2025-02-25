from flask import Flask, request, render_template
import os
from handling import handlers
from flask_cors import CORS
from ollama import OllamaChat

save_text_to_file = handlers.save_text_to_file
transcribe_audio = handlers.transcribe_audio
app = Flask(__name__)
CORS(app)



integer_list = []

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Ollama chat
ollama_chat = OllamaChat()
ollama_chat.preload_model()

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
    
    try:
        response = ollama_chat.chat(message)
        print(f"Ollama response: {response}")  # Debug log
        return {"response": response}
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # Debug log
        return {"error": str(e)}, 500

@app.route("/reset_context", methods=["POST"])
def reset_context():
    response = ollama_chat.reset_context()
    return {"message": response}

@app.route("/get_chat_history")
def get_chat_history():
    return {"history": ollama_chat.context}

if __name__ == "__main__":
    app.run(debug=True, port=9999, host="0.0.0.0")
