import whisper


class handlers():
    def __init__(self,debug=False):
        self.debug = debug


    # debugging still not implemented but at least the functionality is possible now         

    def transcribe_audio(audio_path):
        try:
            model = whisper.load_model("medium", device="cuda")
            result = model.transcribe(audio_path)
            # audio_path = "uploads/recording.wav"  # Change this to your actual file path
            # transcription = transcribe_audio(audio_path)
            return result["text"]
        except Exception as e:
            return f"Error: {str(e)}"
    # transcribe_audio("uploads/recording.wav")


    def save_text_to_file(filename, text):
        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(text)
            return f"Text successfully saved to {filename}"
        except Exception as e:
            return f"Error: {str(e)}"
