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
If you don't know something, say so rather than making up information.

IMPORTANT: When asked about previous conversations after a context reset, do not fabricate memories.
If the context has been reset and you're asked what you remember, clearly state that the conversation
history has been cleared and you don't have access to previous conversations."""
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
        
        # Store the last response
        self.last_response = ""
        
        # Flag to track if the system was recently reset
        self.recently_reset = False
        
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
        
    def chat(self, message, stream=True, context=None):
        """Chat with the model and manage context"""
        # We still calculate importance for logging purposes
        message_importance = self.rag.importance_score(message, role="user")
        print(f"Message importance: {message_importance:.2f}")
        
        # Extract entities from the message
        entities = self.rag.entity_tracker.extract_entities(message)
        if entities:
            print(f"Extracted entities: {len(entities)}")
            for entity in entities[:5]:  # Show top 5 entities
                print(f"- {entity['text']} ({entity['type']})")
        
        # If context is not provided, use the RAG system to find relevant context
        if context is None:
            # Use the new query analysis to make smarter retrieval decisions
            relevant_context = self.rag.search_relevant_context(message, max_results=3)
            
            if relevant_context:
                print(f"Found relevant context for the query:")
                print(relevant_context)
                
                # Add context to the system message with improved instructions
                context_prompt = f"""
I'm providing you with some relevant information from previous conversations with the user:

{relevant_context}

Incorporate this information naturally in your response without explicitly mentioning that it comes from memory or stored information. 
Make your response feel like a natural continuation of the conversation, as if you simply remembered these details.
If the information isn't relevant to the current query, you can ignore it.
"""
            else:
                context_prompt = ""
                print("No relevant context found for this query")
        else:
            # Use the provided context
            context_prompt = f"""
I'm providing you with some relevant information for this conversation:

{context}

Incorporate this information naturally in your response without explicitly mentioning that it comes from memory or stored information. 
Make your response feel like a natural continuation of the conversation, as if you simply remembered these details.
If the information isn't relevant to the current query, you can ignore it.
"""
            print(f"Using provided context: {len(context)} characters")
        
        # Get entity summaries if available and relevant
        entity_context = ""
        query_entities = self.rag.entity_tracker.find_entities_in_query(message)
        if query_entities:
            print(f"Found {len(query_entities)} relevant entities in query:")
            for entity in query_entities:
                print(f"- {entity['text']} ({entity['type']})")
                summary = self.rag.entity_tracker.get_entity_summary(entity['text'])
                if summary:
                    entity_context += f"\nEntity: {entity['text']} - {summary}\n"
        
        # Combine contexts
        if entity_context:
            if context_prompt:
                context_prompt += "\n\nEntity Information:\n" + entity_context
            else:
                context_prompt = "Entity Information:\n" + entity_context
        
        # Prepare messages array with system prompt and context
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # If the system was recently reset and this is a query about memory,
        # add an additional system message to reinforce not fabricating memories
        if self.recently_reset and any(term in message.lower() for term in ["remember", "recall", "memory", "previous"]):
            messages.append({
                "role": "system", 
                "content": "REMINDER: The conversation context was just reset. You have no access to previous conversations. Do not fabricate memories."
            })
        
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
        
        # Store the last response
        self.last_response = response
        
        # Update context with the new exchange
        self.context.append({"role": "user", "content": message})
        self.context.append({"role": "assistant", "content": response})
        
        # Clear the recently reset flag after the first exchange
        self.recently_reset = False
        
        # Manage context window size
        while len(self.context) > self.max_context_length:
            # Remove oldest messages first
            self.context.pop(0)
        
        # Store the user message if important enough
        if message_importance > self.rag.thresholds['storage_min']:
            self.rag.store_message(message, "user", message_importance)
            
            # We no longer store assistant messages as requested
            # Commenting out the storage code
            # if response and len(response) > 50:  # Only store substantial responses
            #     fixed_importance = 0.3  # Just above the storage threshold
            #     self.rag.store_message(response, "assistant", fixed_importance)

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
        try:
            # Clear the conversation context
            self.context = []
            self.last_response = ""
            print("Cleared immediate conversation context")
            
            # Clear the RAG memory collections
            rag_result = self.rag.clear_collection()
            
            # Reset the entity tracker
            self.rag.entity_tracker = self.rag.entity_tracker.__class__()
            print("Reset entity tracker")
            
            # Set the recently reset flag
            self.recently_reset = True
            
            if rag_result:
                return "Both conversation context and long-term memory have been cleared successfully."
            else:
                return "Conversation context cleared, but there was an issue clearing long-term memory."
        except Exception as e:
            error_msg = f"Error resetting context: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg

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
                        
            # Store the full response
            self.last_response = full_response
            return full_response
        except Exception as e:
            error_msg = f"Error in streaming chat: {str(e)}"
            print(error_msg)
            yield error_msg
            
    def get_last_response(self):
        """Get the last response from the assistant"""
        return self.last_response

    def store_interaction(self, user_message, assistant_message):
        """Store both sides of an interaction if they're important enough"""
        # Calculate importance for user message only
        user_importance = self.importance_score(user_message)
        
        # Store user message if important enough
        if user_importance > self.thresholds['storage_min']:
            self.store_message(user_message, "user", user_importance)
            
            # We no longer store assistant messages as requested
            # Commenting out the storage code
            # if assistant_message and len(assistant_message) > 50:  # Only store substantial responses
            #     fixed_importance = 0.3  # Just above the storage threshold
            #     self.store_message(assistant_message, "assistant", fixed_importance)