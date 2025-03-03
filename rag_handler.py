import time
import requests
import re
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import json
import re
import uuid
from entity_tracker import EntityTracker

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
        
        # Initialize entity tracker
        self.entity_tracker = EntityTracker()
        
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
        """Calculate semantic similarity between two texts using embeddings
        
        This uses the sentence transformer model to create embeddings and calculate cosine similarity.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Handle None or empty texts
        if text1 is None or text2 is None:
            return 0.0
            
        text1 = str(text1).strip()
        text2 = str(text2).strip()
        
        if not text1 or not text2:
            return 0.0
            
        try:
            # Generate embeddings for both texts
            embedding1 = self.embed_model.encode(text1, convert_to_tensor=True)
            embedding2 = self.embed_model.encode(text2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            from torch.nn.functional import cosine_similarity
            similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
            
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            print(f"Error calculating semantic similarity: {str(e)}")
            
            # Fall back to simpler method if embedding fails
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
    
    def importance_score(self, text, role="user"):
        """Calculate the importance score of a message based on various factors"""
        if not text or not isinstance(text, str):
            return 0.0
            
        print(f"\n=== IMPORTANCE SCORING ===")
        print(f"1. Evaluating {role} text: {text[:100]}...")
        
        # For assistant messages, return a fixed importance score
        if role == "assistant":
            fixed_importance = 0.3  # Just above the storage threshold
            print(f"2. Fixed importance score for assistant message: {fixed_importance:.2f}")
            return fixed_importance
        
        # Initialize score components
        emotional_content_score = 0.0
        keyword_score = 0.0
        question_presence_score = 0.0
        information_density_score = 0.0
        entity_score = 0.0
        
        # Process text with entity tracker
        important_entities = self.entity_tracker.update_entities(text)
        
        # Calculate entity-based importance score
        if important_entities:
            # Average importance of detected entities
            entity_importance_avg = sum(entity["importance"] for entity in important_entities) / len(important_entities)
            entity_score = entity_importance_avg * 0.4  # Entities contribute up to 40% of total score
            
            print(f"2. Found {len(important_entities)} important entities:")
            for entity in important_entities[:3]:  # Show top 3
                print(f"   - {entity['text']} ({entity['type']}): {entity['importance']:.2f}")
        else:
            print("2. No important entities found")
        
        # Check for emotional content
        emotion_keywords = {
            # Strong positive emotions
            'love': 0.3, 'happy': 0.25, 'excited': 0.25, 'thrilled': 0.3, 'overjoyed': 0.3,
            'ecstatic': 0.3, 'delighted': 0.25, 'grateful': 0.25, 'thankful': 0.25,
            
            # Strong negative emotions
            'angry': 0.3, 'sad': 0.25, 'depressed': 0.3, 'anxious': 0.3, 'worried': 0.25,
            'scared': 0.3, 'terrified': 0.3, 'frustrated': 0.25, 'upset': 0.25, 'hurt': 0.25,
            'disappointed': 0.25, 'heartbroken': 0.3, 'devastated': 0.3, 'miserable': 0.3,
            
            # Moderate emotions
            'good': 0.15, 'bad': 0.15, 'fine': 0.1, 'okay': 0.1, 'content': 0.15,
            'annoyed': 0.15, 'irritated': 0.15, 'concerned': 0.15, 'nervous': 0.15,
            
            # Emotional states
            'feeling': 0.1, 'feel': 0.1, 'felt': 0.1, 'emotion': 0.1, 'mood': 0.1,
            'stress': 0.2, 'anxiety': 0.2, 'depression': 0.2, 'trauma': 0.2,
            
            # Intensifiers that suggest emotional content
            'very': 0.05, 'really': 0.05, 'extremely': 0.1, 'incredibly': 0.1,
            'so': 0.05, 'too': 0.05, 'absolutely': 0.1
        }
        
        # Split text into words once for efficiency
        words = text.split()
        
        # Check for emotion words in the text
        emotion_score = 0
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in emotion_keywords:
                emotion_score += emotion_keywords[word_lower]
        
        # Cap emotion score at 0.2
        emotional_content_score = min(emotion_score, 0.2)
        print(f"3. Emotional content score: {emotional_content_score:.2f}")
        
        # Check for key content indicators
        important_keywords = {
            'important': 0.1, 'remember': 0.1, 'key': 0.08, 
            'must': 0.08, 'critical': 0.1, 'essential': 0.08,
            'note': 0.06, 'significant': 0.08, 'urgent': 0.1,
            'priority': 0.1, 'crucial': 0.1
        }
        keyword_score_raw = sum(important_keywords.get(word.lower().strip('.,!?;:'), 0) for word in words)
        keyword_score = min(keyword_score_raw, 0.15)
        print(f"4. Keyword score: {keyword_score:.2f}")
        
        # Check for question presence
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who']
        if any(indicator in text.lower() for indicator in question_indicators):
            question_presence_score = 0.15
        print(f"5. Question presence score: {question_presence_score:.2f}")
        
        # Check for information density
        # Check for numbers, dates, proper nouns (capitalized words)
        info_indicators = sum([
            len([w for w in words if any(c.isdigit() for c in w)]) * 0.05,  # Numbers
            len([w for w in words if w and w[0].isupper()]) * 0.03,  # Proper nouns
            1 if any(month in text.lower() for month in ['january', 'february', 'march', 
                    'april', 'may', 'june', 'july', 'august', 'september', 
                    'october', 'november', 'december']) else 0  # Dates
        ])
        information_density_score = min(info_indicators, 0.1)
        print(f"6. Information density score: {information_density_score:.2f}")
        
        # Calculate total score with entity component
        total_score = emotional_content_score + keyword_score + question_presence_score + information_density_score + entity_score
        
        # Ensure score is between 0 and 1
        total_score = min(1.0, total_score)
        
        print(f"7. Final importance score: {total_score:.2f}")
        return total_score
    
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
        
        # Calculate importance for user message
        user_importance = self.importance_score(user_message, role="user")
        
        # Store user message if important enough
        if user_importance > self.thresholds['storage_min']:
            user_id = self.store_message(user_message, "user", user_importance)
            
            # Store assistant message with a fixed importance score
            if assistant_message and len(assistant_message) > 50:  # Only store substantial responses
                fixed_importance = 0.3  # Just above the storage threshold
                assistant_id = self.store_message(assistant_message, "assistant", fixed_importance)
                
                if user_id and assistant_id:
                    print(f"Stored complete interaction: User ({user_id}) and Assistant ({assistant_id})")
                    return True
        
        return False
    
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
            
    def clear_collection(self):
        """Clear all memory collections (long-term, short-term, and ephemeral)"""
        try:
            print("\n=== CLEARING ALL MEMORY COLLECTIONS ===")
            
            # Get all IDs from long-term memory
            long_term_results = self.long_term_memory.get()
            if long_term_results and 'ids' in long_term_results and long_term_results['ids']:
                self.long_term_memory.delete(ids=long_term_results['ids'])
                print(f"Cleared {len(long_term_results['ids'])} items from long-term memory")
            
            # Get all IDs from short-term memory
            short_term_results = self.short_term_memory.get()
            if short_term_results and 'ids' in short_term_results and short_term_results['ids']:
                self.short_term_memory.delete(ids=short_term_results['ids'])
                print(f"Cleared {len(short_term_results['ids'])} items from short-term memory")
            
            # Clear ephemeral memory
            ephemeral_count = len(self.ephemeral_memory)
            self.ephemeral_memory = []
            print(f"Cleared {ephemeral_count} items from ephemeral memory")
            
            # Clear storage buffer
            buffer_count = len(self.storage_buffer)
            self.storage_buffer = []
            print(f"Cleared {buffer_count} items from storage buffer")
            
            print("=== MEMORY COLLECTIONS CLEARED SUCCESSFULLY ===\n")
            return True
        except Exception as e:
            print(f"Error clearing collections: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def categorize_memory(self, text, importance, source_type=""):
        """Determine which type of memory should store this information
        
        Args:
            text (str): The text content to categorize
            importance (float): The calculated importance score
            source_type (str): The source type of the content (e.g., "transcription_audio")
            
        Returns:
            str: The memory type to store in ("long_term", "short_term", or "ephemeral")
        """
        # Check for important entities in the text
        important_entities = []
        if hasattr(self, 'entity_tracker'):
            # Extract entities from the text
            entities = self.entity_tracker.extract_entities(text)
            # Find entities with high importance
            for entity in entities:
                entity_key = entity["text"].lower()
                if entity_key in self.entity_tracker.entities:
                    entity_data = self.entity_tracker.entities[entity_key]
                    # If entity is important or mentioned multiple times
                    if entity_data["importance"] > 0.7 or entity_data["mentions"] >= 3:
                        important_entities.append(entity)
        
        # If text contains important entities, store in long-term memory
        if important_entities and len(important_entities) > 0:
            print(f"   Found {len(important_entities)} important entities - storing in long-term memory")
            return "long_term"
            
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
        # Lower the threshold for long-term memory from default
        if importance > 0.6:  # Lowered from self.thresholds["short_term_max"]
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

    def extract_keywords(self, text):
        """Extract important keywords from text for filtering.
        
        Args:
            text (str): The text to extract keywords from
            
        Returns:
            list: List of important keywords
        """
        # Handle None or empty text
        if text is None or not text.strip():
            return []
            
        # Simple stopword list
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of', 
                    'that', 'this', 'these', 'those', 'it', 'its', 'have', 'has', 'had',
                    'do', 'does', 'did', 'am', 'be', 'been', 'being', 'as', 'if', 'then',
                    'so', 'than', 'such', 'both', 'each', 'few', 'more', 'most', 'some',
                    'will', 'would', 'should', 'can', 'could', 'may', 'might', 'must'}
        
        # Split text into words and convert to lowercase
        words = [word.lower() for word in text.split()]
        
        # Remove stopwords and short words
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = [word for word in keywords if not (word in seen or seen.add(word))]
        
        return unique_keywords

    def search_relevant_context(self, query, max_results=3, force_retrieval=False):
        """Search for relevant context based on the query"""
        if not query or not isinstance(query, str):
            return ""
            
        print(f"\n=== CONTEXT SEARCH ===")
        print(f"1. Query: {query}")
        
        # Analyze query to determine if retrieval is needed
        if not force_retrieval:
            analysis = self.analyze_query(query)
            if not analysis["retrieval_recommended"]:
                print(f"2. Retrieval not recommended: {analysis['reason']}")
                return ""
        
        # Extract entities from the query
        query_entities = self.entity_tracker.find_entities_in_query(query)
        
        # Extract keywords for filtering
        keywords = self.extract_keywords(query)
        
        print(f"2. Extracted keywords: {keywords}")
        if query_entities:
            print(f"3. Found {len(query_entities)} relevant entities in query:")
            for entity in query_entities:
                print(f"   - {entity['text']} ({entity['type']}): {entity['importance']:.2f}")
        
        # Prepare results container
        all_results = []
        
        # Search in long-term memory
        try:
            # Generate query embedding
            query_embedding = self.embed_model.encode([query])[0].tolist()
            
            # Search in long-term memory
            results = self.long_term_memory.query(
                query_embeddings=[query_embedding],
                n_results=max_results
            )
            
            if results and 'documents' in results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'][0] else {}
                    
                    # Calculate semantic similarity score
                    semantic_score = self.semantic_similarity(query, doc)
                    
                    # Calculate keyword match score
                    doc_keywords = self.extract_keywords(doc)
                    keyword_matches = sum(1 for k in keywords if k in doc_keywords)
                    keyword_score = min(1.0, keyword_matches / max(1, len(keywords)))
                    
                    # Calculate entity match score
                    entity_score = 0.0
                    if query_entities:
                        # Check if any entities from the query appear in this document
                        entity_matches = 0
                        for entity in query_entities:
                            if entity['text'].lower() in doc.lower():
                                entity_matches += 1
                        entity_score = min(1.0, entity_matches / len(query_entities))
                    
                    # Combined score (60% semantic, 20% keyword, 20% entity)
                    combined_score = (semantic_score * 0.6) + (keyword_score * 0.2) + (entity_score * 0.2)
                    
                    if combined_score > 0.5:  # Only include if combined score is above threshold
                        all_results.append((combined_score, doc, metadata, 
                                          semantic_score, keyword_score, entity_score))
        except Exception as e:
            print(f"Error searching long-term memory: {e}")
        
        # Search in short-term memory
        try:
            # Search in short-term memory
            results = self.short_term_memory.query(
                query_embeddings=[query_embedding],
                n_results=max_results
            )
            
            if results and 'documents' in results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'][0] else {}
                    
                    # Calculate semantic similarity score
                    semantic_score = self.semantic_similarity(query, doc)
                    
                    # Calculate keyword match score
                    doc_keywords = self.extract_keywords(doc)
                    keyword_matches = sum(1 for k in keywords if k in doc_keywords)
                    keyword_score = min(1.0, keyword_matches / max(1, len(keywords)))
                    
                    # Calculate entity match score
                    entity_score = 0.0
                    if query_entities:
                        # Check if any entities from the query appear in this document
                        entity_matches = 0
                        for entity in query_entities:
                            if entity['text'].lower() in doc.lower():
                                entity_matches += 1
                        entity_score = min(1.0, entity_matches / len(query_entities))
                    
                    # Combined score (60% semantic, 20% keyword, 20% entity)
                    combined_score = (semantic_score * 0.6) + (keyword_score * 0.2) + (entity_score * 0.2)
                    
                    if combined_score > 0.5:  # Only include if combined score is above threshold
                        all_results.append((combined_score, doc, metadata, 
                                          semantic_score, keyword_score, entity_score))
        except Exception as e:
            print(f"Error searching short-term memory: {e}")
        
        # Search in ephemeral memory
        for memory in self.ephemeral_memory:
            if 'text' in memory:
                # Calculate semantic similarity score
                semantic_score = self.semantic_similarity(query, memory['text'])
                
                # Calculate keyword match score
                doc_keywords = self.extract_keywords(memory['text'])
                keyword_matches = sum(1 for k in keywords if k in doc_keywords)
                keyword_score = min(1.0, keyword_matches / max(1, len(keywords)))
                
                # Calculate entity match score
                entity_score = 0.0
                if query_entities:
                    # Check if any entities from the query appear in this document
                    entity_matches = 0
                    for entity in query_entities:
                        if entity['text'].lower() in memory['text'].lower():
                            entity_matches += 1
                    entity_score = min(1.0, entity_matches / len(query_entities))
                
                # Combined score (60% semantic, 20% keyword, 20% entity)
                combined_score = (semantic_score * 0.6) + (keyword_score * 0.2) + (entity_score * 0.2)
                
                if combined_score > 0.5:  # Only include if combined score is above threshold
                    all_results.append((combined_score, memory['text'], memory.get('metadata', {}), 
                                      semantic_score, keyword_score, entity_score))
        
        if not all_results:
            print("4. No relevant memories found")
            return ""
            
        # Sort by combined score (highest first)
        all_results.sort(reverse=True, key=lambda x: x[0])
        
        # Take top results
        top_results = all_results[:max_results]
        
        print(f"4. Found {len(top_results)} relevant memories")
        
        # Format the results
        formatted_results = []
        for i, (combined_score, text, metadata, semantic_score, keyword_score, entity_score) in enumerate(top_results):
            # Truncate long results
            if len(text) > 300:
                text = text[:297] + "..."
            
            # Add source information if available
            source_info = ""
            if metadata and 'source' in metadata:
                source_info = f" (Source: {metadata['source']})"
            
            # Add score information
            score_info = f"[Relevance: {combined_score:.2f}, Semantic: {semantic_score:.2f}, Keyword: {keyword_score:.2f}, Entity: {entity_score:.2f}]"
            
            formatted_results.append(f"Memory {i+1}{source_info}: {text}\n{score_info}\n")
        
        # Join the formatted results
        result_text = "\n".join(formatted_results)
        
        print(f"5. Returning {len(formatted_results)} memories")
        return result_text

    def store_transcription(self, text, source="transcription", processed_text=None, role="user"):
        """Store a transcription in the appropriate memory store
        
        Args:
            text (str): The original transcription text
            source (str): The source of the transcription
            processed_text (str, optional): Processed version of the text (e.g., key points)
            role (str): The role of the message sender ('user' or 'assistant')
            
        Returns:
            bool: True if stored, False otherwise
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            print(f"Skipping transcription storage: Text too short or invalid")
            return False
            
        print(f"\n=== TRANSCRIPTION STORAGE ===")
        print(f"1. Processing transcription (length: {len(text.split())} words)")
        
        # Calculate importance based on the original transcription
        importance = self.importance_score(text, role=role)
        print(f"2. Importance score: {importance:.2f}")
        
        # Skip if below threshold
        if importance < self.thresholds["storage_min"]:
            print(f"3. Below storage threshold ({self.thresholds['storage_min']}), discarding")
            return False
        
        # Use processed text if provided, otherwise use original
        storage_text = processed_text if processed_text else text
        
        # Determine memory category based on importance
        memory_type = self.categorize_memory(storage_text, importance, source_type=source)
        print(f"4. Categorized as {memory_type} memory")
        
        # Generate metadata
        metadata = {
            "source": source,
            "timestamp": time.time(),
            "importance": importance,
            "type": memory_type,
            "role": role
        }
        
        # Generate tags
        tags = self.generate_tags(storage_text)
        if tags:
            # Convert tags list to a comma-separated string for ChromaDB compatibility
            metadata["tags"] = ",".join(tags)
            print(f"5. Generated tags: {', '.join(tags)}")
        
        # Store based on memory type
        if memory_type == "ephemeral":
            # Store in ephemeral memory (in-memory only)
            memory_id = str(uuid.uuid4())
            self.ephemeral_memory.append({
                "id": memory_id,
                "text": storage_text,
                "metadata": metadata
            })
            print(f"6. Stored in ephemeral memory with ID: {memory_id}")
            return True
            
        elif memory_type == "short_term":
            # Store in short-term memory
            try:
                memory_id = str(uuid.uuid4())
                self.short_term_memory.add(
                    ids=[memory_id],
                    documents=[storage_text],
                    metadatas=[metadata]
                )
                print(f"6. Stored in short-term memory with ID: {memory_id}")
                return True
            except Exception as e:
                print(f"Error storing in short-term memory: {e}")
                return False
                
        elif memory_type == "long_term":
            # Store in long-term memory
            try:
                memory_id = str(uuid.uuid4())
                self.long_term_memory.add(
                    ids=[memory_id],
                    documents=[storage_text],
                    metadatas=[metadata],
                    embeddings=[self.embed_model.encode(storage_text).tolist()]
                )
                print(f"6. Stored in long-term memory with ID: {memory_id}")
                return True
            except Exception as e:
                print(f"Error storing in long-term memory: {e}")
                return False
        
        return False

    def analyze_query(self, query):
        """Analyze a query to determine if retrieval would be valuable.
        
        This function examines the query for specificity, ambiguity, and potential
        for finding relevant information in the knowledge base.
        
        Args:
            query (str): The user's query
            
        Returns:
            dict: Analysis results including retrieval_recommended (bool) and reason
        """
        if not query or len(query.strip()) < 3:
            return {"retrieval_recommended": False, "reason": "Query too short"}
            
        # 1. Check for question indicators (suggests information seeking)
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'did', 'do', 'is', 'are', 'can']
        is_question = any(indicator in query.lower().split() or indicator == '?' and '?' in query for indicator in question_indicators)
        
        # 2. Check for specific entities or proper nouns (suggests specific information need)
        words = query.split()
        proper_nouns = [w for w in words if w and w[0].isupper() and not w.isupper()]
        has_proper_nouns = len(proper_nouns) > 0
        
        # 3. Check for temporal references (suggests time-based information need)
        temporal_indicators = ['yesterday', 'today', 'tomorrow', 'last', 'next', 'previous', 'upcoming', 
                              'week', 'month', 'year', 'monday', 'tuesday', 'wednesday', 'thursday', 
                              'friday', 'saturday', 'sunday', 'january', 'february', 'march', 'april',
                              'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
        has_temporal = any(indicator in query.lower() for indicator in temporal_indicators)
        
        # 4. Check for memory-related terms (suggests explicit memory recall)
        memory_indicators = ['remember', 'recall', 'mentioned', 'said', 'told', 'talked', 'discussed', 'noted', 'recorded']
        has_memory_terms = any(indicator in query.lower() for indicator in memory_indicators)
        
        # 5. Check for personal references (suggests personal information need)
        personal_indicators = ['my', 'i', 'me', 'mine', 'we', 'our', 'us', 'ours']
        has_personal_refs = any(indicator in query.lower().split() for indicator in personal_indicators)
        
        # Combine factors to make a decision
        retrieval_score = sum([
            1.0 if is_question else 0.0,
            0.7 if has_proper_nouns else 0.0,
            0.8 if has_temporal else 0.0,
            0.9 if has_memory_terms else 0.0,
            0.6 if has_personal_refs else 0.0
        ])
        
        # Determine if retrieval is recommended
        retrieval_recommended = retrieval_score >= 1.0  # At least one strong indicator
        
        # Determine the primary reason
        reason = "General query"
        if is_question:
            reason = "Information-seeking question"
        elif has_memory_terms:
            reason = "Explicit memory recall request"
        elif has_temporal:
            reason = "Time-based information need"
        elif has_proper_nouns:
            reason = "Contains specific entities"
        elif has_personal_refs:
            reason = "Contains personal references"
            
        return {
            "retrieval_recommended": retrieval_recommended,
            "reason": reason,
            "score": retrieval_score,
            "factors": {
                "is_question": is_question,
                "has_proper_nouns": has_proper_nouns,
                "has_temporal": has_temporal,
                "has_memory_terms": has_memory_terms,
                "has_personal_refs": has_personal_refs
            }
        }

    def store_message(self, message, role, importance):
        """Store a single message in the appropriate memory store
        
        Args:
            message (str): The message text to store
            role (str): The role of the message sender ('user' or 'assistant')
            importance (float): The importance score of the message
            
        Returns:
            str: The ID of the stored document, or None if not stored
        """
        if not message or not isinstance(message, str) or len(message) < 5:
            print(f"Skipping storage: Message too short or invalid")
            return None
            
        print(f"\n=== MESSAGE STORAGE ===")
        print(f"1. Processing {role} message (length: {len(message.split())} words)")
        
        # Extract key points if the message is long enough
        if len(message) > self.thresholds["key_points_min_length"]:
            print(f"2. Extracting key points from message")
            key_points = self.extract_key_points(message, source=role)
            if key_points is None:
                print("   No key insights found, using original")
                key_points = message
        else:
            print(f"2. Message too short, using as is")
            key_points = message
        
        # Check similarity with existing memories
        all_memories = []
        
        # Get all memories from collections
        try:
            long_term_results = self.long_term_memory.get()
            if long_term_results and 'documents' in long_term_results:
                for doc in long_term_results['documents']:
                    all_memories.append({"text": doc})
                    
            short_term_results = self.short_term_memory.get()
            if short_term_results and 'documents' in short_term_results:
                for doc in short_term_results['documents']:
                    all_memories.append({"text": doc})
        except Exception as e:
            print(f"Error retrieving memories for similarity check: {e}")
        
        # Add ephemeral memories
        all_memories.extend(self.ephemeral_memory)
        
        if all_memories:
            highest_similarity = 0
            similar_text = ""
            
            for memory in all_memories:
                if 'text' not in memory:
                    continue
                    
                similarity = self.semantic_similarity(key_points, memory['text'])
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    similar_text = memory['text']
                
                if similarity > self.thresholds["similarity_max"]:  # High similarity threshold
                    print(f"3. Discarding due to high similarity ({similarity:.2f}) with existing memory")
                    print(f"   Similar to: {similar_text[:50]}...")
                    return None
            
            print(f"4. Highest similarity with existing memories: {highest_similarity:.2f}")
        else:
            print("4. No existing memories to compare with")
        
        # Determine memory category based on importance
        memory_type = self.categorize_memory(key_points, importance, source_type=role)
        print(f"5. Categorized as {memory_type} memory (importance: {importance:.2f})")
        
        # Generate metadata
        metadata = {
            "source": role,
            "timestamp": time.time(),
            "importance": importance,
            "type": memory_type
        }
        
        # Generate tags
        tags = self.generate_tags(key_points)
        if tags:
            # Convert tags list to a comma-separated string for ChromaDB compatibility
            metadata["tags"] = ",".join(tags)
            print(f"6. Generated tags: {', '.join(tags)}")
        
        # Store based on memory type
        if memory_type == "ephemeral":
            # Store in ephemeral memory (in-memory only)
            memory_id = str(uuid.uuid4())
            self.ephemeral_memory.append({
                "id": memory_id,
                "text": key_points,
                "metadata": metadata
            })
            print(f"7. Stored in ephemeral memory with ID: {memory_id}")
            return memory_id
            
        elif memory_type == "short_term":
            # Store in short-term memory
            try:
                memory_id = str(uuid.uuid4())
                self.short_term_memory.add(
                    ids=[memory_id],
                    documents=[key_points],
                    metadatas=[metadata]
                )
                print(f"7. Stored in short-term memory with ID: {memory_id}")
                return memory_id
            except Exception as e:
                print(f"Error storing in short-term memory: {e}")
                return None
                
        elif memory_type == "long_term":
            # Store in long-term memory
            try:
                memory_id = str(uuid.uuid4())
                self.long_term_memory.add(
                    ids=[memory_id],
                    documents=[key_points],
                    metadatas=[metadata],
                    embeddings=[self.embed_model.encode(key_points).tolist()]
                )
                print(f"7. Stored in long-term memory with ID: {memory_id}")
                return memory_id
            except Exception as e:
                print(f"Error storing in long-term memory: {e}")
                return None
        
        return None