import whisper

def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("medium")
        result = model.transcribe(audio_path)
        # audio_path = "uploads/recording.wav"  # Change this to your actual file path
        # transcription = transcribe_audio(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error: {str(e)}"

# transcribe_audio("uploads/recording.wav")