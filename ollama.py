import json
import requests
import numpy as np
from rag_handler import RAGHandler

class OllamaChat:
    """Chat with Ollama models with RAG support"""
    
    def __init__(self, base_url="http://localhost:11434", model="deepseek-r1", 
                 system_prompt=None, options=None, max_context_length=10, rag_config=None):
        """Initialize the chat client
        
        Args:
            base_url (str): Base URL for the Ollama API
            model (str): Model name to use
            system_prompt (str): System prompt to use
            options (dict): Model options
            max_context_length (int): Maximum number of messages to keep in context
            rag_config (dict): Configuration for the RAG system
        """
        self.base_url = base_url
        self.model = model
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant. Answer questions concisely and accurately.
If you don't know something, say so rather than making up information."""
        self.system_prompt = system_prompt
        
        # Default options if none provided
        if options is None:
            options = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192  # Increased context window for deepseek-r1
            }
        self.options = options
        
        # Context management
        self.context = []
        self.max_context_length = max_context_length
        
        # Initialize RAG with configuration
        self.rag = RAGHandler(config=rag_config)
        
        print(f"Initialized OllamaChat with model: {model}")
        print(f"RAG system initialized with configuration: {rag_config if rag_config else 'default'}")
        
    def add_to_context(self, role, text):
        """Add a message to the context"""
        self.context.append({"role": role, "content": text})
        
        # Manage context window size
        while len(self.context) > self.max_context_length:
            self.context.pop(0)
            
        print(f"Added to context: {text[:50]}...")
        
    def estimate_tokens(self, text):
        """Estimate the number of tokens in a text"""
        # Rough estimate: 4 characters per token
        return len(text) // 4
        
    def manage_context_window(self):
        """Manage context window based on importance and recency"""
        if not self.context:
            return
            
        # Calculate total tokens
        total_tokens = sum(self.estimate_tokens(msg["content"]) for msg in self.context)
        
        while len(self.context) > self.max_context_length or total_tokens > self.max_context_tokens:
            # Keep the first message (system context) if it exists
            if self.context[0]["role"] == "system" and len(self.context) > 1:
                self.context.pop(1)
            else:
                self.context.pop(0)
            # Recalculate tokens
            total_tokens = sum(self.estimate_tokens(msg["content"]) for msg in self.context)
        
    def chat(self, message, stream=True):
        """Chat with the model and manage context"""
        # Evaluate message importance for RAG operations
        message_importance = self.rag.importance_score(message)
        print(f"Message importance: {message_importance:.2f}")
        
        # Get relevant context using our new search method
        relevant_context = self.rag.search_relevant_context(message, max_results=3)
        
        if relevant_context:
            print(f"Found relevant context for the query:")
            print(relevant_context)
            
            # Add context to the system message
            context_prompt = f"""
I'm providing you with some relevant information from my memory that might help answer the user's question:

{relevant_context}

Use this information if it's relevant to the user's query. If it's not relevant, simply ignore it.
"""
        else:
            context_prompt = ""
            print("No relevant context found for this query")
        
        # Prepare messages array with system prompt and context
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add context if available
        if context_prompt:
            messages.append({"role": "system", "content": context_prompt})
        
        # Add conversation context (recent messages)
        messages.extend(self.context)
        
        # Add the current message
        messages.append({"role": "user", "content": message})
        
        # Call the Ollama API
        response = ""
        if stream:
            # Stream the response
            for chunk in self._stream_chat(messages):
                response += chunk
                yield chunk
        else:
            # Get the full response at once
            response = self._chat(messages)
            yield response
        
        # Update context with the new exchange
        self.context.append({"role": "user", "content": message})
        self.context.append({"role": "assistant", "content": response})
        
        # Manage context window size
        while len(self.context) > self.max_context_length:
            # Remove oldest messages first
            self.context.pop(0)
        
        # Store the interaction in RAG if important enough
        self.rag.store_interaction(message, response)
        
    def _chat(self, messages):
        """Send a chat request to Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": self.options
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"Error in chat: {e}")
            return f"I encountered an error: {str(e)}"

    def reset_context(self):
        """Reset both immediate context and RAG memory"""
        self.context = []
        self.rag.clear_collection()
        return "Both conversation context and long-term memory have been cleared."

    def format_context(self, relevant_docs):
        """Format retrieved context for the LLM with improved structure"""
        if not relevant_docs:
            return ""
        
        context = "Important context from previous conversations:\n\n"
        
        # Group similar contexts together
        for i, doc in enumerate(relevant_docs, 1):
            context += f"[Memory {i}] {doc}\n"
            
        context += "\nInstructions for using context:"
        context += "\n- Reference these memories if they are relevant to the current question"
        context += "\n- Ignore memories that aren't directly relevant"
        context += "\n- If you use a memory, briefly mention which one you're drawing from"
        context += "\n- Maintain conversation flow even when using memories"
        
        return context

    def _stream_chat(self, messages):
        """Stream a chat response from Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": self.options
                },
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line.decode('utf-8'))
                        if 'message' in chunk_data and 'content' in chunk_data['message']:
                            chunk = chunk_data['message']['content']
                            full_response += chunk
                            yield chunk
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {line}")
                        
            return full_response
        except Exception as e:
            error_msg = f"Error in streaming chat: {str(e)}"
            print(error_msg)
            yield error_msg