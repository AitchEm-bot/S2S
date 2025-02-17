import requests
import json

class OllamaChat:
    def __init__(self, model="mistral"):
        self.model = model
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
        # Include all previous context plus the new message
        messages = self.context + [{"role": "user", "content": message}]
        
        try:
            print(f"Sending request to Ollama with {len(messages)} messages")  # Debug log
            print(f"Context being sent: {json.dumps(messages, indent=2)}")  # Debug log
            
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "model": self.model,
                    "messages": messages,
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
                            # Concatenate the chunks
                            assistant_message += response_json["message"]["content"]
                            print(f"Received chunk: {response_json['message']['content']}")  # Debug log
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {str(e)}, line: {line}")  # Debug log
                
                if assistant_message:
                    # Add the assistant's response to the context
                    self.context.append({"role": "assistant", "content": assistant_message})
                    return assistant_message
                else:
                    return "No response received from Ollama"
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                print(error_msg)  # Debug log
                return error_msg
                
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to Ollama. Is it running?"
            print(error_msg)  # Debug log
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(error_msg)  # Debug log
            return error_msg

    def reset_context(self):
        """Reset the conversation context"""
        self.context = []
        return "Context has been reset"
