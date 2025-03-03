import whisper


class handlers():
    def __init__(self,debug=False):
        self.debug = debug


    # debugging still not implemented but at least the functionality is possible now         

    def transcribe_audio(audio_path):
        try:
            print(f"\n=== AUDIO RECEIVED ===")
            print(f"Transcribing audio file: {audio_path}")
            
            # Check if CUDA is available
            import torch
            if torch.cuda.is_available():
                print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
                device = "cuda"
            else:
                print("CUDA is not available. Using CPU instead.")
                device = "cpu"
                
            # Load the model with the appropriate device
            model = whisper.load_model("medium", device=device)
            print(f"Whisper model loaded on {device}, starting transcription...")
            
            # Add map_location parameter when transcribing to handle any device mismatches
            result = model.transcribe(audio_path)
            print(f"Transcription completed successfully")
            return result["text"]
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            # Provide more helpful error message
            if "CUDA" in str(e):
                return f"Error: CUDA issue detected. {str(e)}. Try restarting the application or check your GPU drivers."
            return f"Error: {str(e)}"
    # transcribe_audio("uploads/recording.wav")


    def save_text_to_file(filename, text):
        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(text)
            return f"Text successfully saved to {filename}"
        except Exception as e:
            return f"Error: {str(e)}"
