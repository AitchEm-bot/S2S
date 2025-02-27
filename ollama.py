import requests
import json
from rag_handler import RAGHandler

class OllamaChat:
    def __init__(self, model_name="mistral"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        self.context = []
        self.rag = RAGHandler()
        
    def add_to_context(self, text):
        """Add transcribed text to context"""
        self.context.append({
            "role": "user", 
            "content": text,
            "source": "transcription"  # Mark this as a transcription
        })
        print(f"Added to context: {text}")
        
    def chat(self, message):
        """Send message to Ollama with context and RAG"""
        try:
            # Store user message in RAG first
            print(f"\n1. Storing message in RAG: {message}")
            self.rag.store_interaction(message)
            
            # Add to context
            self.context.append({"role": "user", "content": message})
            
            # Get relevant context from previous conversations
            print("\n2. Getting relevant context for:", message)
            relevant_context = self.rag.get_relevant_context(message)
            context_prompt = self.rag.format_context(relevant_docs=relevant_context)
            
            # Include all previous context plus the new message
            messages = []
            if context_prompt:
                messages.append({"role": "system", "content": context_prompt})
                print("\n3. Added system context:", context_prompt)
            
            messages.extend(self.context)
            
            print("\n4. Full message list being sent to Ollama:")
            for idx, msg in enumerate(messages, 1):
                print(f"\n   Message {idx}:")
                print(f"   Role: {msg['role']}")
                print(f"   Content: {msg['content'][:100]}..." if len(msg['content']) > 100 else f"   Content: {msg['content']}")
            
            print(f"\nTotal messages being sent: {len(messages)}")
            
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "model": self.model_name,
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
                            if "message" in response_json:
                                assistant_message += response_json["message"]["content"]
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {str(e)}, line: {line}")
                
                if assistant_message:
                    assistant_message = assistant_message.lstrip(' ')
                    self.context.append({"role": "assistant", "content": assistant_message})
                    return assistant_message
                
                return "No response received from Ollama"
                
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                print(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            print(error_msg)
            return error_msg

    def reset_context(self):
        """Reset both immediate context and RAG memory"""
        self.context = []
        self.rag.clear_collection()
        return "Both conversation context and long-term memory have been cleared."