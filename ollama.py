import requests
import json

class OllamaChat:
    def __init__(self, model_name="mistral"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        self.context = []
        
    def add_to_context(self, text):
        """Add transcribed text to context"""
        self.context.append({
            "role": "user", 
            "content": text,
            "source": "transcription"  # Mark this as a transcription
        })
        print(f"Added to context: {text}")
        
    def chat(self, message):
        """Send message to Ollama with context"""
        # Add the user message to context first
        self.context.append({"role": "user", "content": message})
        
        try:
            print(f"Sending request to Ollama with {len(self.context)} messages")
            
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "model": self.model_name,
                    "messages": self.context,
                    "stream": True
                },
                stream=True
            )
            
            if response.status_code == 200:
                assistant_message = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            response_json = json.loads(line.decode('utf-8'))
                            assistant_message += response_json["message"]["content"]
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {str(e)}, line: {line}")
                
                if assistant_message:
                    # Remove only the first space if it exists
                    assistant_message = assistant_message.lstrip(' ')
                    # Add the assistant's response to the context
                    self.context.append({"role": "assistant", "content": assistant_message})
                    return assistant_message
                else:
                    return "No response received from Ollama"
                
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(error_msg)
            return error_msg

    def reset_context(self):
        """Reset the conversation context"""
        self.context = []
        return "Context has been reset"