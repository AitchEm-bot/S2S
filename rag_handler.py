import time
import requests
import re
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import json
import re
import uuid

class RAGHandler:
    def __init__(self, config=None):
        # Default thresholds
        self.thresholds = {
            "storage_min": 0.3,           # Minimum score to store anything
            "ephemeral_max": 0.4,         # Maximum score for ephemeral memory
            "short_term_max": 0.7,        # Maximum score for short-term memory
            "retrieval_min": 0.3,         # Minimum query importance for retrieval
            "similarity_max": 0.8,        # Maximum similarity to consider duplicate
            "ephemeral_similarity": 0.5,  # Minimum similarity for ephemeral retrieval
            "key_points_min_length": 30,  # Minimum length to trigger key points extraction
            "short_term_expiry_days": 7,  # Days before short-term memory expires
            "skip_maintenance": False     # Flag to skip maintenance on startup
        }
        
        # Override defaults with provided config
        if config:
            self.thresholds.update(config)

        self.base_url = "http://localhost:11434"

        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collections for different memory types
        self.long_term_memory = self.chroma_client.get_or_create_collection("long_term_memory")
        self.short_term_memory = self.chroma_client.get_or_create_collection("short_term_memory")
        
        # For backward compatibility
        self.collection = self.long_term_memory
        
        # Load the embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.storage_buffer = []
        self.last_store_time = time.time()
        
        # Ephemeral memory (cleared on restart)
        self.ephemeral_memory = []
        
        # Run memory maintenance on startup (unless skipped)
        if not self.thresholds.get("skip_maintenance", False):
            self.maintain_memory()
        else:
            print("Skipping memory maintenance on startup (as configured)")
        
    def maintain_memory(self):
        """Perform maintenance tasks on memory stores"""
        print("\n=== MEMORY MAINTENANCE ===")
        
        # 1. Expire old short-term memories
        self.expire_short_term_memories()
        
        # 2. Clean up ephemeral memory (just in case)
        if len(self.ephemeral_memory) > 100:  # Arbitrary limit
            print(f"Pruning ephemeral memory from {len(self.ephemeral_memory)} items")
            # Keep only the most recent 50 items
            self.ephemeral_memory = self.ephemeral_memory[-50:]
            
        # 3. Consolidate memories
        self.consolidate_memories()
        
        print("=== MEMORY MAINTENANCE COMPLETE ===\n")
        
    def consolidate_memories(self):
        """Review short-term memories and promote important ones to long-term memory"""
        try:
            print("Starting memory consolidation...")
            
            # Get all short-term memories
            short_term_results = self.short_term_memory.get()
            
            # Check if short_term_results exists and has the expected structure
            if not short_term_results or not isinstance(short_term_results, dict):
                print("No short-term memories to consolidate (empty results)")
                return
                
            # Check if the required fields exist
            if 'documents' not in short_term_results or not short_term_results['documents']:
                print("No short-term memories to consolidate (no documents)")
                return
                
            if 'metadatas' not in short_term_results or not short_term_results['metadatas']:
                print("No short-term memories to consolidate (no metadata)")
                return
                
            if 'embeddings' not in short_term_results or not short_term_results['embeddings']:
                print("No short-term memories to consolidate (no embeddings)")
                return
                
            if 'ids' not in short_term_results or not short_term_results['ids']:
                print("No short-term memories to consolidate (no ids)")
                return
                
            # Track items to promote and their IDs for deletion from short-term
            to_promote = []
            ids_to_delete = []
            
            # Check each memory for promotion
            for i, text in enumerate(short_term_results['documents']):
                metadata = short_term_results['metadatas'][i]
                embedding = short_term_results['embeddings'][i]
                
                # Re-evaluate importance with potentially new criteria
                importance = self.importance_score(text)
                
                # Check if memory has become more important or has been accessed frequently
                access_count = metadata.get('access_count', 0)
                
                # Promotion criteria: high importance or frequently accessed
                if importance > self.thresholds["short_term_max"] or access_count >= 3:
                    print(f"Promoting memory to long-term: '{text[:50]}...' (importance: {importance:.2f}, accesses: {access_count})")
                    
                    # Add to promotion list
                    to_promote.append({
                        "text": text,
                        "embedding": embedding,
                        "metadata": metadata,
                        "id": short_term_results['ids'][i]
                    })
                    
                    ids_to_delete.append(short_term_results['ids'][i])
            
            # Promote memories to long-term
            if to_promote:
                for item in to_promote:
                    # Update metadata to indicate promotion
                    item["metadata"]["promoted_from"] = "short_term"
                    item["metadata"]["promotion_time"] = time.time()
                    
                    # Add to long-term memory
                    self.long_term_memory.add(
                        embeddings=[item["embedding"]],
                        documents=[item["text"]],
                        ids=[item["id"]],
                        metadatas=[item["metadata"]]
                    )
                
                # Remove promoted items from short-term memory
                self.short_term_memory.delete(ids=ids_to_delete)
                print(f"Promoted {len(to_promote)} memories from short-term to long-term")
            else:
                print("No memories qualified for promotion")
                
        except Exception as e:
            print(f"Error during memory consolidation: {e}")
            import traceback
            traceback.print_exc()
            
    def expire_short_term_memories(self):
        """Remove expired items from short-term memory"""
        try:
            # Get all short-term memories
            short_term_results = self.short_term_memory.get()
            
            # Check if short_term_results exists and has the expected structure
            if not short_term_results or not isinstance(short_term_results, dict):
                print("No short-term memories to check for expiration (empty results)")
                return
                
            # Check if the required fields exist
            if 'ids' not in short_term_results or not short_term_results['ids']:
                print("No short-term memories to check for expiration (no ids)")
                return
                
            if 'metadatas' not in short_term_results or not short_term_results['metadatas']:
                print("No short-term memories to check for expiration (no metadata)")
                return
                
            current_time = time.time()
            expiry_seconds = self.thresholds["short_term_expiry_days"] * 24 * 60 * 60
            ids_to_delete = []
            
            # Check each memory for expiration
            for i, metadata in enumerate(short_term_results['metadatas']):
                # Skip if no timestamp (for backward compatibility)
                if 'timestamp' not in metadata:
                    continue
                    
                memory_age = current_time - metadata['timestamp']
                if memory_age > expiry_seconds:
                    ids_to_delete.append(short_term_results['ids'][i])
            
            # Delete expired memories
            if ids_to_delete:
                self.short_term_memory.delete(ids=ids_to_delete)
                print(f"Expired {len(ids_to_delete)} items from short-term memory")
            else:
                print("No expired short-term memories found")
                
        except Exception as e:
            print(f"Error during short-term memory expiration: {e}")
            import traceback
            traceback.print_exc()
            
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts
        
        This is a simple implementation using word overlap.
        In a production system, you would use embeddings for better semantic matching.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
            
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def extract_key_points(self, text, source="chat"):
        """Extract key points from text using LLM
        
        Args:
            text (str): The text to extract key points from
            source (str): The source of the text (chat or transcription)
            
        Returns:
            str: The extracted key points, or None if no key points were found
        """
        try:
            # Filter out content between <think> tags
            filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            
            # If the filtered text is empty, return None
            if not filtered_text.strip():
                return None
                
            # Check if text is too short (using word count)
            word_count = len(filtered_text.split())
            if word_count < 5:  # Minimum word threshold
                print(f"Text too short for key point extraction ({word_count} words)")
                return None
                
            # Always use mistral model
            model = "mistral"
            
            # Different prompts based on source
            if source == "transcription" and "audio" in source:
                # Special prompt for audio journal entries
                prompt = f"""
You are analyzing a personal audio journal entry. Extract the most valuable information about the user's:
1. Emotional state and feelings
2. Key insights or realizations
3. Significant information about their life, preferences, or thoughts

Respond with ONLY a single concise sentence summarizing the key information that would be most valuable to remember about the user.
If no meaningful insights can be found, respond with ONLY the text "NO_KEY_INSIGHTS" and nothing else.
IMPORTANT: When talking about the journal entry refer to the speaker as "the user".

Journal entry: {filtered_text}
"""
            elif source == "transcription":
                # Regular transcription prompt
                prompt = f"""
Extract ONLY the key details that would be useful for future reference from this transcription.
Respond with ONLY a single concise sentence summarizing the key information.
If no meaningful insights can be found, respond with ONLY the text "NO_KEY_INSIGHTS" and nothing else.

Transcription: {filtered_text}
"""
            else:
                # Chat prompt
                prompt = f"""
Extract ONLY the key details that would be useful for future reference from this message.
Respond with ONLY a single concise sentence summarizing the key information.
If no meaningful insights can be found, respond with ONLY the text "NO_KEY_INSIGHTS" and nothing else.

Message: {filtered_text}
"""
            
            # Call the Ollama API
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You extract key points from text."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                },
                timeout=30
            )
            
            response.raise_for_status()
            key_points = response.json()["message"]["content"].strip()
            
            # Filter out any thinking content in the response
            key_points = re.sub(r'<think>.*?</think>', '', key_points, flags=re.DOTALL).strip()
            
            # Check if the response is "NO_KEY_INSIGHTS" or contains it
            if key_points == "NO_KEY_INSIGHTS" or "NO_KEY_INSIGHTS" in key_points:
                print("No key insights found in the text")
                return None
                
            return key_points
            
        except Exception as e:
            print(f"Error extracting key points: {e}")
            return None
    
    def importance_score(self, text):
        """Calculate importance score of a message using multiple factors"""
        score = 0.0
        
        # Split text into words once for efficiency
        words = text.split()
        word_count = len(words)
        
        # 1. Length factor (max 0.3) - now explicitly using word count
        length_score = min(word_count / 50.0, 1.0) * 0.3
        score += length_score
        
        # 2. Key content indicators (max 0.3)
        important_keywords = {
            'important': 0.1, 'remember': 0.1, 'key': 0.08, 
            'must': 0.08, 'critical': 0.1, 'essential': 0.08,
            'note': 0.06, 'significant': 0.08
        }
        keyword_score = sum(weight for word, weight in important_keywords.items() 
                           if word in text.lower())
        score += min(keyword_score, 0.3)
        
        # 3. Question presence (max 0.2)
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who']
        if any(indicator in text.lower() for indicator in question_indicators):
            score += 0.2
        
        # 4. Information density (max 0.2)
        # Check for numbers, dates, proper nouns (capitalized words)
        info_indicators = sum([
            len([w for w in words if any(c.isdigit() for c in w)]) * 0.05,  # Numbers
            len([w for w in words if w and w[0].isupper()]) * 0.03,  # Proper nouns
            1 if any(month in text.lower() for month in ['january', 'february', 'march', 
                    'april', 'may', 'june', 'july', 'august', 'september', 
                    'october', 'november', 'december']) else 0  # Dates
        ])
        score += min(info_indicators, 0.2)
        
        print(f"Importance score breakdown:")
        print(f"- Length score: {length_score:.2f} (based on {word_count} words)")
        print(f"- Keyword score: {keyword_score:.2f}")
        print(f"- Question presence: {0.2 if any(indicator in text.lower() for indicator in question_indicators) else 0:.2f}")
        print(f"- Information density: {min(info_indicators, 0.2):.2f}")
        print(f"Final score: {min(score, 1.0):.2f}")
        
        return min(score, 1.0)  # Ensure score doesn't exceed 1.0
    
    def should_store(self, new_info):
        """Determine if new information should be stored"""
        print(f"\n=== STORAGE EVALUATION ===")
        print(f"1. Evaluating text: {new_info[:100]}...")
        
        # Check importance first
        importance = self.importance_score(new_info)
        if importance < self.thresholds["storage_min"]:
            print(f"2. Message importance {importance:.2f} below threshold, discarding")
            return False
            
        # Check similarity with existing memories
        all_results = self.collection.get()
        if 'metadatas' in all_results and all_results['metadatas']:
            highest_similarity = 0
            similar_text = ""
            
            for metadata in all_results['metadatas']:
                similarity = self.semantic_similarity(new_info, metadata['text'])
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    similar_text = metadata['text']
                
                if similarity > self.thresholds["similarity_max"]:  # High similarity threshold
                    print(f"3. Discarding due to high similarity ({similarity:.2f}) with existing memory")
                    print(f"   Similar to: {similar_text[:100]}...")
                    return False
            
            print(f"3. Highest similarity with existing memories: {highest_similarity:.2f}")
        else:
            print("3. No existing memories to compare with")
        
        # Extract key points if text is long
        if len(new_info.split()) > self.thresholds["key_points_min_length"]:
            print("4. Text is long, extracting key points")
            key_points = self.extract_key_points(new_info)
            
            if key_points is None:
                print("5. No key insights found, discarding")
                return False
                
            if key_points != new_info:  # If extraction succeeded
                print("5. Storing extracted key points instead of full text")
                # Replace the storage buffer entry with the key points
                self.storage_buffer.append({
                    "text": key_points,
                    "type": "key_points",
                    "original_length": len(new_info.split()),
                    "summary_length": len(key_points.split())
                })
                return True
        
        print(f"4. Message passed all filters, will store")
        return True
    
    def store_interaction(self, user_message, assistant_message):
        """Store a user-assistant interaction in the appropriate memory store"""
        # Skip if either message is empty
        if not user_message or not assistant_message:
            print("Skipping storage: Empty message")
            return
            
        print(f"\n=== INTERACTION STORAGE ATTEMPT ===")
        
        # Extract key points from user message using mistral model
        if len(user_message) > self.thresholds["key_points_min_length"]:
            print(f"1. Extracting key points from user message (length: {len(user_message.split())})")
            user_key_points = self.extract_key_points(user_message, source="chat") 
            if user_key_points is None:
                print("   No key insights found in user message, using original")
                user_key_points = user_message
        else:
            print(f"1. User message too short ({len(user_message.split())} words), using as is")
            user_key_points = user_message
        
        # Evaluate only the user's message for storage decision
        print(f"\n=== STORAGE EVALUATION ===")
        print(f"2. Evaluating user message importance")
        
        # Check importance of user message only
        importance = self.importance_score(user_key_points)
        if importance < self.thresholds["storage_min"]:
            print(f"3. User message importance {importance:.2f} below threshold {self.thresholds['storage_min']}, discarding")
            print("Skipping storage: Not important enough")
            return
            
        # Check similarity with existing memories
        all_memories = []
        
        # Combine all memory stores for similarity check
        if hasattr(self, 'long_term_store') and self.long_term_store:
            all_memories.extend(self.long_term_store)
            
        if hasattr(self, 'short_term_store') and self.short_term_store:
            all_memories.extend(self.short_term_store)
            
        if hasattr(self, 'ephemeral_store') and self.ephemeral_store:
            all_memories.extend(self.ephemeral_store)
            
        if all_memories:
            highest_similarity = 0
            similar_text = ""
            
            for memory in all_memories:
                if 'text' not in memory:
                    continue
                    
                similarity = self.semantic_similarity(user_key_points, memory['text'])
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    similar_text = memory['text']
                
                if similarity > self.thresholds["similarity_max"]:  # High similarity threshold
                    print(f"4. Discarding due to high similarity ({similarity:.2f}) with existing memory")
                    print(f"   Similar to: {similar_text[:100]}...")
                    print("Skipping storage: Too similar to existing memories")
                    return
            
            print(f"4. Highest similarity with existing memories: {highest_similarity:.2f}")
        else:
            print("4. No existing memories to compare with")
        
        # If we get here, we should store the interaction
        # Extract key points from assistant message for storage
        if len(assistant_message) > self.thresholds["key_points_min_length"]:
            print(f"5. Extracting key points from assistant message (length: {len(assistant_message.split())})")
            assistant_key_points = self.extract_key_points(assistant_message, source="chat")
            if assistant_key_points is None:
                print("   No key insights found in assistant message, using original")
                assistant_key_points = assistant_message
        else:
            print(f"5. Assistant message too short ({len(assistant_message.split())} words), using as is")
            assistant_key_points = assistant_message
        
        # Create combined text with key points for storage
        combined_text = f"User: {user_key_points}\nAssistant: {assistant_key_points}"
        
        # Generate tags for the interaction
        tags = self.generate_tags(combined_text)
        
        # Store the document
        doc_id = self.store_document(combined_text, source="interaction")
        
        print(f"6. Stored interaction with ID: {doc_id}")
        return doc_id
    
    def generate_tags(self, text):
        """Generate tags for a memory based on content analysis"""
        try:
            # Default tags
            tags = ["conversation"]
            
            # Check for question patterns
            if re.search(r'\?|how|what|why|when|where|who|which|can you|could you', text.lower()):
                tags.append("question")
                
            # Check for code or technical content
            if re.search(r'```|def |class |function|import |from |var |const |let |public |private', text):
                tags.append("code")
                
            # Check for explanations
            if re.search(r'explain|explanation|describe|definition|concept|mean[s]?|understand', text.lower()):
                tags.append("explanation")
                
            # Check for instructions or how-to
            if re.search(r'how to|steps|guide|tutorial|instructions|procedure', text.lower()):
                tags.append("how-to")
                
            # Check for examples
            if re.search(r'example|instance|sample|illustration', text.lower()):
                tags.append("example")
                
            # Check for errors or issues
            if re.search(r'error|issue|problem|bug|fix|resolve|exception|fail', text.lower()):
                tags.append("troubleshooting")
                
            # Check for definitions
            if re.search(r'define|definition|meaning|what is|what are', text.lower()):
                tags.append("definition")
                
            # Check for comparisons
            if re.search(r'versus|vs|compare|comparison|difference|better|worse|pros and cons', text.lower()):
                tags.append("comparison")
                
            # Check for personal information
            if re.search(r'my name|my email|my phone|my address|my account|my password', text.lower()):
                tags.append("personal")
                
            return list(set(tags))  # Remove duplicates
            
        except Exception as e:
            print(f"Error generating tags: {e}")
            return ["conversation"]  # Default tag
            
    def get_memories_by_tag(self, tag, max_results=10):
        """Retrieve memories that match a specific tag"""
        try:
            print(f"Retrieving memories with tag: {tag}")
            
            # Search in long-term memory
            long_term_results = self.long_term_memory.get(
                where={"tags": {"$contains": tag}}
            )
            
            # Search in short-term memory
            short_term_results = self.short_term_memory.get(
                where={"tags": {"$contains": tag}}
            )
            
            # Combine results
            combined_results = []
            
            # Process long-term results
            if long_term_results and 'metadatas' in long_term_results and long_term_results['metadatas']:
                for i, metadata in enumerate(long_term_results['metadatas']):
                    combined_results.append({
                        "text": long_term_results['documents'][i],
                        "source": "long_term",
                        "metadata": metadata,
                        "id": long_term_results['ids'][i]
                    })
            
            # Process short-term results
            if short_term_results and 'metadatas' in short_term_results and short_term_results['metadatas']:
                for i, metadata in enumerate(short_term_results['metadatas']):
                    combined_results.append({
                        "text": short_term_results['documents'][i],
                        "source": "short_term",
                        "metadata": metadata,
                        "id": short_term_results['ids'][i]
                    })
            
            # Search in ephemeral memory
            for item in self.ephemeral_memory:
                if "metadata" in item and "tags" in item["metadata"] and tag in item["metadata"]["tags"]:
                    combined_results.append({
                        "text": item["text"],
                        "source": "ephemeral",
                        "metadata": item["metadata"],
                        "id": str(hash(item["text"]))
                    })
            
            # Sort by timestamp (newest first)
            combined_results.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
            
            # Take top results
            return combined_results[:max_results]
            
        except Exception as e:
            print(f"Error retrieving memories by tag: {e}")
            return []
            
    def get_all_tags(self):
        """Get a list of all tags used in the system"""
        try:
            all_tags = set()
            
            # Get tags from long-term memory
            long_term_results = self.long_term_memory.get()
            if long_term_results and 'metadatas' in long_term_results and long_term_results['metadatas']:
                for metadata in long_term_results['metadatas']:
                    if "tags" in metadata:
                        all_tags.update(metadata["tags"])
            
            # Get tags from short-term memory
            short_term_results = self.short_term_memory.get()
            if short_term_results and 'metadatas' in short_term_results and short_term_results['metadatas']:
                for metadata in short_term_results['metadatas']:
                    if "tags" in metadata:
                        all_tags.update(metadata["tags"])
            
            # Get tags from ephemeral memory
            for item in self.ephemeral_memory:
                if "metadata" in item and "tags" in item["metadata"]:
                    all_tags.update(item["metadata"]["tags"])
            
            return sorted(list(all_tags))
            
        except Exception as e:
            print(f"Error retrieving all tags: {e}")
            return []
    
    def categorize_memory(self, text, importance, source_type=""):
        """Determine which type of memory should store this information
        
        Args:
            text (str): The text content to categorize
            importance (float): The calculated importance score
            source_type (str): The source type of the content (e.g., "transcription_audio")
            
        Returns:
            str: The memory type to store in ("long_term", "short_term", or "ephemeral")
        """
        # Special categorization for audio journal entries
        if "transcription_audio" in source_type:
            # Audio journals: ephemeral = 0.0, short-term = 0.3-0.5, long-term = >0.5
            if importance > 0.5:
                return "long_term"
            elif importance >= 0.3:
                return "short_term"
            else:
                return "ephemeral"
        
        # Standard categorization for other content types
        # Long-term memory: high importance, contains key facts
        if importance > self.thresholds["short_term_max"]:
            return "long_term"
            
        # Short-term memory: medium importance, might be useful for a while
        elif importance > self.thresholds["ephemeral_max"]:
            return "short_term"
            
        # Ephemeral memory: low importance, only useful for current session
        else:
            return "ephemeral"

    def _flush_buffer(self):
        """Flush storage buffer to persistent storage using memory categorization"""
        if not self.storage_buffer:
            print("Buffer empty, nothing to flush")
            return
        
        print(f"\n=== FLUSHING BUFFER ===")
        print(f"1. Buffer size: {len(self.storage_buffer)}")
        
        long_term_count = 0
        short_term_count = 0
        ephemeral_count = 0
        
        for item in self.storage_buffer:
            try:
                # Handle both dictionary items and string items (for backward compatibility)
                if isinstance(item, dict):
                    text = item["text"]
                    metadata = item
                else:
                    text = item
                    metadata = {"text": text, "type": "legacy_message"}
                
                # Check for valid text
                if not text or not isinstance(text, str):
                    print(f"Skipping invalid item: {text}")
                    continue
                
                # Get the source type for categorization
                source_type = metadata.get("type", "")
                
                # Check if this is an audio transcription and use journal_importance_score if it is
                if "transcription_audio" in source_type:
                    importance = self.journal_importance_score(text)
                    print(f"Using journal importance score for audio transcription: {importance:.2f}")
                else:
                    importance = self.importance_score(text)
                
                # Store in appropriate memory based on the calculated importance and source type
                memory_type = self.categorize_memory(text, importance, source_type)
                
                print(f"Storing in {memory_type} memory (importance: {importance:.2f}, source: {source_type})")
                
                # Store in the appropriate memory type
                if memory_type == "long_term":
                    # Store in long-term memory (ChromaDB)
                    embedding = self.embed_model.encode([text])[0].tolist()
                    self.long_term_memory.add(
                        embeddings=[embedding],
                        ids=[str(hash(text))],
                        metadatas=[metadata]
                    )
                    long_term_count += 1
                elif memory_type == "short_term":
                    # Store in short-term memory (ChromaDB with TTL)
                    embedding = self.embed_model.encode([text])[0].tolist()
                    self.short_term_memory.add(
                        embeddings=[embedding],
                        ids=[str(hash(text))],
                        metadatas=[metadata]
                    )
                    short_term_count += 1
                else:
                    # Store in ephemeral memory (in-memory only)
                    self.ephemeral_memory.append({
                        "text": text,
                        "metadata": metadata,
                        "timestamp": metadata['timestamp']
                    })
                    ephemeral_count += 1
                
            except Exception as e:
                print(f"Error processing item: {str(e)}")
                continue
        
        print(f"2. Memory distribution:")
        print(f"   - Long-term: {long_term_count}")
        print(f"   - Short-term: {short_term_count}")
        print(f"   - Ephemeral: {ephemeral_count}")
        
        # Clear buffer regardless of storage success
        self.storage_buffer = []
        self.last_store_time = time.time()
        print("3. Buffer cleared and timestamp updated")
        
    def store_in_memory(self, text, metadata=None):
        """Store information in the appropriate memory store"""
        if metadata is None:
            metadata = {"text": text, "type": "regular_message"}
        
        # Add timestamp if not present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = time.time()
        
        # Get the source type for categorization
        source_type = metadata.get("type", "")
        
        # Check if this is an audio transcription and use journal_importance_score if it is
        if "transcription_audio" in source_type:
            importance = self.journal_importance_score(text)
            print(f"Using journal importance score for audio transcription: {importance:.2f}")
        else:
            importance = self.importance_score(text)
        
        # Store in appropriate memory based on the calculated importance and source type
        memory_type = self.categorize_memory(text, importance, source_type)
        
        print(f"Storing in {memory_type} memory (importance: {importance:.2f}, source: {source_type})")
        
        if memory_type == "long_term":
            # Store in long-term memory (ChromaDB)
            embedding = self.embed_model.encode([text])[0].tolist()
            self.long_term_memory.add(
                embeddings=[embedding],
                ids=[str(hash(text))],
                metadatas=[metadata]
            )
            return "long_term"
            
        elif memory_type == "short_term":
            # Store in short-term memory (ChromaDB with TTL)
            embedding = self.embed_model.encode([text])[0].tolist()
            self.short_term_memory.add(
                embeddings=[embedding],
                ids=[str(hash(text))],
                metadatas=[metadata]
            )
            return "short_term"
            
        else:
            # Store in ephemeral memory (in-memory only)
            self.ephemeral_memory.append({
                "text": text,
                "metadata": metadata,
                "timestamp": metadata['timestamp']
            })
            return "ephemeral"

    def journal_importance_score(self, text):
        """Calculate importance score of a journal entry using specialized factors
        
        This method is specifically designed for audio journal entries, which are
        considered more valuable for understanding the user's personal context.
        """
        score = 0.0
        
        # Split text into words once for efficiency
        words = text.split()
        word_count = len(words)
        
        # 1. Length factor (max 0.25) - now explicitly using word count
        # Journal entries are valuable even when shorter
        length_score = min(word_count / 40.0, 1.0) * 0.25
        score += length_score
        
        # 2. Emotional content indicators (max 0.3)
        emotional_keywords = {
            'feel': 0.05, 'felt': 0.05, 'feeling': 0.05, 
            'happy': 0.05, 'sad': 0.05, 'angry': 0.05, 'anxious': 0.05,
            'excited': 0.05, 'worried': 0.05, 'stressed': 0.05,
            'love': 0.05, 'hate': 0.05, 'afraid': 0.05, 'hopeful': 0.05,
            'disappointed': 0.05, 'proud': 0.05, 'grateful': 0.05,
            'frustrated': 0.05, 'overwhelmed': 0.05
        }
        emotion_score = sum(weight for word, weight in emotional_keywords.items() 
                           if word in text.lower())
        score += min(emotion_score, 0.3)
        
        # 3. Personal experience indicators (max 0.25)
        personal_indicators = {
            'i ': 0.02, 'me': 0.02, 'my': 0.02, 'mine': 0.02, 'myself': 0.02,
            'we': 0.02, 'us': 0.02, 'our': 0.02, 'ours': 0.02, 'ourselves': 0.02,
            'today': 0.03, 'yesterday': 0.03, 'tomorrow': 0.03,
            'think': 0.03, 'thought': 0.03, 'believe': 0.03,
            'want': 0.03, 'need': 0.03, 'hope': 0.03,
            'tried': 0.03, 'trying': 0.03, 'attempt': 0.03
        }
        personal_score = sum(weight for word, weight in personal_indicators.items() 
                           if f" {word} " in f" {text.lower()} ")
        score += min(personal_score, 0.25)
        
        # 4. Significant life events (max 0.2)
        life_event_indicators = [
            r'\b(birth|death|wedding|funeral|graduation|anniversary)\b',
            r'\b(job|career|promotion|fired|quit|hired|interview)\b',
            r'\b(move|moving|moved|relocation|relocate|relocated)\b',
            r'\b(relationship|dating|marriage|divorce|breakup|engaged)\b',
            r'\b(health|sick|illness|disease|diagnosis|surgery|hospital)\b',
            r'\b(travel|trip|vacation|holiday|journey|adventure)\b',
            r'\b(goal|achievement|accomplishment|milestone|success|failure)\b',
            r'\b(decision|choice|choose|chose|decide|decided)\b'
        ]
        
        life_event_score = 0
        for pattern in life_event_indicators:
            if re.search(pattern, text.lower()):
                life_event_score += 0.05
                
        score += min(life_event_score, 0.2)
        
        print(f"Journal importance score breakdown:")
        print(f"- Length score: {length_score:.2f} (based on {word_count} words)")
        print(f"- Emotional content score: {min(emotion_score, 0.3):.2f}")
        print(f"- Personal experience score: {min(personal_score, 0.25):.2f}")
        print(f"- Life event score: {min(life_event_score, 0.2):.2f}")
        print(f"Final journal score: {min(score, 1.0):.2f}")
        
        # Journal entries are inherently more valuable, so we add a base boost
        # This ensures they're more likely to be stored
        boosted_score = min(score + 0.15, 1.0)
        
        if boosted_score > score:
            print("--------------------------------")
            print(f"Applied journal entry boost: +0.15 → Final score: {boosted_score:.2f}")
            print("--------------------------------")
            
        return boosted_score  # Ensure score doesn't exceed 1.0

    def store_document(self, text, source="unknown"):
        """Store a document in the database with appropriate metadata.
        
        Args:
            text (str): The text content to store
            source (str): The source of the document (e.g., "transcription_audio", "chat")
            
        Returns:
            str: The document ID if stored successfully, None otherwise
        """
        if not text:
            return None
            
        # Create metadata
        metadata = {
            "text": text,
            "type": source,
            "length": len(text),
            "word_count": len(text.split()),
            "timestamp": time.time()
        }
        
        # Add to buffer
        self.storage_buffer.append(metadata)
        print("=== BUFFERING ===")
        print(f"Added to buffer. Buffer size: {len(self.storage_buffer)}")
        
        # Generate a document ID
        doc_id = f"{source}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Store if buffer is full or enough time has passed
        current_time = time.time()
        if len(self.storage_buffer) >= 5 or (current_time - self.last_store_time) >= 300:
            print("Triggering buffer flush...")
            self._flush_buffer()
            print("Buffer flushed to memory stores")
        else:
            print(f"Keeping in buffer. Will flush when full or after timeout")
            print(f"Time since last flush: {current_time - self.last_store_time:.1f}s")
        
        return doc_id

    def search_relevant_context(self, query, max_results=3):
        """Search for relevant information in the database based on the user's query.
        
        Args:
            query (str): The user's query or message
            max_results (int): Maximum number of results to return
            
        Returns:
            str: A concatenated string of relevant information, or empty string if none found
        """
        if not query or len(query.strip()) < 5:
            return ""
            
        print(f"Searching for context relevant to: '{query[:50]}...' if longer")
        
        # Combine all memory stores for search
        all_memories = []
        
        # Add long-term memories
        if hasattr(self, 'long_term_store') and self.long_term_store:
            all_memories.extend(self.long_term_store)
            
        # Add short-term memories
        if hasattr(self, 'short_term_store') and self.short_term_store:
            all_memories.extend(self.short_term_store)
            
        # Add ephemeral memories (if they exist)
        if hasattr(self, 'ephemeral_store') and self.ephemeral_store:
            all_memories.extend(self.ephemeral_store)
            
        if not all_memories:
            print("No memories available to search")
            return ""
            
        print(f"Searching across {len(all_memories)} total memories")
        
        # Simple keyword matching for now
        # In a production system, you would use embeddings and semantic search
        query_words = set(query.lower().split())
        scored_results = []
        
        for memory in all_memories:
            if 'text' not in memory:
                continue
                
            memory_text = memory['text'].lower()
            # Count matching words
            match_count = sum(1 for word in query_words if word in memory_text)
            if match_count > 0:
                # Score based on match count and recency
                recency_factor = 1.0
                if 'timestamp' in memory:
                    age_hours = (time.time() - memory['timestamp']) / 3600
                    recency_factor = max(0.5, 1.0 - (age_hours / 720))  # Decay over ~30 days
                
                score = match_count * recency_factor
                scored_results.append((score, memory['text']))
        
        # Sort by score (highest first)
        scored_results.sort(reverse=True)
        
        # Take top results
        top_results = [text for _, text in scored_results[:max_results]]
        
        if not top_results:
            print("No relevant memories found")
            return ""
            
        print(f"Found {len(top_results)} relevant memories")
        
        # Format the results
        formatted_results = []
        for i, result in enumerate(top_results):
            # Truncate long results
            if len(result) > 300:
                result = result[:297] + "..."
            formatted_results.append(f"[Memory {i+1}]: {result}")
            
        return "\n\n".join(formatted_results)

    def store_transcription(self, text, source="transcription"):
        """Store a transcription in the database with appropriate metadata.
        
        Args:
            text (str): The transcription text to store
            source (str): The source of the transcription (e.g., "audio", "video")
            
        Returns:
            str: The document ID if stored successfully, None otherwise
        """
        if not text:
            print("Empty transcription, not storing")
            return None
        
        # Create the full source type first to avoid reference errors
        full_source = f"transcription_{source}"
        
        # Print detailed information about the transcription
        word_count = len(text.split())
        print("\n=== TRANSCRIPTION DETAILS ===")
        print(f"Source: {source}")
        print(f"Word count: {word_count}")
        print(f"Original text: \"{text[:150]}{'...' if len(text) > 150 else ''}\"")
        print("--------------------------------")

        # Try to extract key points if text is long enough
        if word_count >= 5:
            print("\nAttempting to extract key points...")
            key_points = self.extract_key_points(text, source=full_source)
            if key_points:
                print(f"Extracted key points: \"{key_points}\"")
            else:
                print("No key points extracted")
        else:
            print("\nText too short for key point extraction")
        
        # Evaluation section
        print("=== EVALUATING ===")
        
        # Calculate importance score before storing
        if source == "audio":
            importance = self.journal_importance_score(text)
            print(f"Journal importance score: {importance:.2f}")
        else:
            importance = self.importance_score(text)
            print(f"Standard importance score: {importance:.2f}")
        
        # Store the transcription using the document store method
        doc_id = self.store_document(text, source=full_source)
        
        return doc_id