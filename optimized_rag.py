"""
Optimized RAG Handler

This module implements the optimized RAG system as described in updates.md:
1. Store raw user data without summarization
2. Use hybrid search (FAISS + BM25)
3. Summarize only after retrieval
"""

import time
import numpy as np
import faiss
import re
import threading
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import requests
import json
from functools import lru_cache
import os
import pickle

# Import the original RAG handler to extend it
from rag_handler import RAGHandler

class OptimizedRAGHandler(RAGHandler):
    """Optimized RAG handler that implements the improvements from updates.md"""
    
    def __init__(self, config=None):
        # Initialize the parent class
        super().__init__(config)
        
        # Initialize BM25 for keyword search
        self.bm25 = None
        self.stored_entries = []
        self.entry_ids = []
        
        # Flag to store raw data
        self.store_raw_data = True
        
        # Flag to use hybrid search
        self.use_hybrid_search = True
        
        # Flag to summarize after retrieval
        self.summarize_after_retrieval = True
        
        # Initialize FAISS index for KNN search
        self.embedding_dimension = 384  # Default for sentence-transformers models
        self.knn_index = faiss.IndexFlatL2(self.embedding_dimension)
        self.embeddings = []
        
        # Initialize BM25 with empty corpus
        self._initialize_bm25()
        
        print("Optimized RAG handler initialized with improvements from updates.md")
    
    def _initialize_bm25(self):
        """Initialize BM25 with the current stored entries"""
        if not self.stored_entries:
            # Get all stored entries from ChromaDB
            try:
                # Get entries from long-term memory
                long_term_results = self.long_term_memory.get()
                if long_term_results and 'documents' in long_term_results:
                    for i, doc in enumerate(long_term_results['documents']):
                        # Create entry with metadata
                        entry = {
                            "text": doc,
                            "id": long_term_results['ids'][i] if 'ids' in long_term_results else f"lt_{i}",
                            "timestamp": time.time() - (len(long_term_results['documents']) - i) * 3600,  # Approximate timestamps
                            "importance": 0.5,  # Default importance
                            "role": "system"
                        }
                        self.stored_entries.append(entry)
                        if 'ids' in long_term_results:
                            self.entry_ids.append(long_term_results['ids'][i])
                        
                        # Extract embedding if available
                        if 'embeddings' in long_term_results and long_term_results['embeddings']:
                            embedding = np.array(long_term_results['embeddings'][i]).astype('float32')
                            self.embeddings.append(embedding)
                            # Add to KNN index
                            if len(embedding) == self.embedding_dimension:
                                self.knn_index.add(np.array([embedding]))
                
                # Get entries from short-term memory
                short_term_results = self.short_term_memory.get()
                if short_term_results and 'documents' in short_term_results:
                    for i, doc in enumerate(short_term_results['documents']):
                        # Create entry with metadata
                        entry = {
                            "text": doc,
                            "id": short_term_results['ids'][i] if 'ids' in short_term_results else f"st_{i}",
                            "timestamp": time.time() - i * 60,  # More recent timestamps for short-term memory
                            "importance": 0.7,  # Higher default importance for short-term
                            "role": "system"
                        }
                        self.stored_entries.append(entry)
                        if 'ids' in short_term_results:
                            self.entry_ids.append(short_term_results['ids'][i])
                        
                        # Extract embedding if available
                        if 'embeddings' in short_term_results and short_term_results['embeddings']:
                            embedding = np.array(short_term_results['embeddings'][i]).astype('float32')
                            self.embeddings.append(embedding)
                            # Add to KNN index
                            if len(embedding) == self.embedding_dimension:
                                self.knn_index.add(np.array([embedding]))
            except Exception as e:
                print(f"Error initializing stored entries: {e}")
        
        # Create tokenized corpus for BM25
        tokenized_corpus = [entry["text"].split() if isinstance(entry, dict) else entry.split() for entry in self.stored_entries]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        
        print(f"BM25 initialized with {len(self.stored_entries)} entries")
        print(f"KNN index initialized with {self.knn_index.ntotal} vectors")
    
    def store_message(self, message, role, importance):
        """Store a message with its raw content and update BM25 and KNN indexes"""
        # Store the message using the parent method
        message_id = super().store_message(message, role, importance)
        
        if message_id:
            # Add to stored entries for BM25
            # Store as a dictionary with metadata
            entry = {
                "text": message,
                "id": message_id,
                "timestamp": time.time(),  # Add timestamp for recency ranking
                "importance": importance,  # Store importance for importance ranking
                "role": role
            }
            self.stored_entries.append(entry)
            self.entry_ids.append(message_id)
            
            # Generate embedding for KNN
            try:
                embedding = self.embed_model.encode(message).astype('float32')
                self.embeddings.append(embedding)
                
                # Add to KNN index
                self.knn_index.add(np.array([embedding]))
                
                print(f"Added message to BM25 and KNN indexes (total entries: {len(self.stored_entries)})")
            except Exception as e:
                print(f"Error adding to KNN index: {e}")
            
            # Update BM25
            tokenized_corpus = [entry["text"].split() if isinstance(entry, dict) else entry.split() for entry in self.stored_entries]
            self.bm25 = BM25Okapi(tokenized_corpus)
        
        return message_id
    
    def _search_relevant_context_impl(self, query, max_results=3):
        """Implementation of hybrid search using both KNN and BM25"""
        if not self.use_hybrid_search or (not self.bm25 and self.knn_index.ntotal == 0):
            # Fall back to original implementation if hybrid search is disabled or no data
            return super()._search_relevant_context_impl(query, max_results)
        
        print(f"Using hybrid search (KNN + BM25) for query: {query}")
        
        # Results containers
        semantic_results = []
        keyword_results = []
        
        # 1. KNN Search (Semantic)
        try:
            if self.knn_index.ntotal > 0:
                # Generate query embedding
                query_embedding = self.embed_model.encode(query).astype('float32')
                
                # Reshape for FAISS
                query_vector = query_embedding.reshape(1, -1)
                
                # Perform KNN search
                k = min(max_results, self.knn_index.ntotal)
                distances, indices = self.knn_index.search(query_vector, k)
                
                # Get the corresponding entries
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.stored_entries):
                        entry = self.stored_entries[idx]
                        distance = distances[0][i]
                        semantic_results.append({
                            "entry": entry,
                            "text": entry["text"] if isinstance(entry, dict) else entry,
                            "score": 1.0 / (1.0 + distance),  # Convert distance to similarity score
                            "source": "knn"
                        })
                
                print(f"KNN found {len(semantic_results)} relevant results")
        except Exception as e:
            print(f"Error in KNN search: {e}")
        
        # 2. BM25 Search (Keyword)
        try:
            if self.bm25:
                # Get top results from BM25
                tokenized_query = query.split()
                
                # Create corpus of just the text for BM25
                corpus = [entry["text"] if isinstance(entry, dict) else entry for entry in self.stored_entries]
                
                # Get BM25 scores
                bm25_scores = self.bm25.get_scores(tokenized_query)
                
                # Get top indices
                top_indices = np.argsort(bm25_scores)[-max_results:][::-1]
                
                # Get the corresponding entries
                for idx in top_indices:
                    if idx < len(self.stored_entries) and bm25_scores[idx] > 0:
                        entry = self.stored_entries[idx]
                        score = bm25_scores[idx]
                        keyword_results.append({
                            "entry": entry,
                            "text": entry["text"] if isinstance(entry, dict) else entry,
                            "score": score,
                            "source": "bm25"
                        })
                
                print(f"BM25 found {len(keyword_results)} relevant results")
        except Exception as e:
            print(f"Error in BM25 search: {e}")
        
        # 3. Combine and rank results with recency and importance
        combined_results = semantic_results + keyword_results
        
        # Calculate current time for recency scoring
        current_time = time.time()
        
        # Normalize scores and add recency and importance factors
        for result in combined_results:
            entry = result["entry"]
            
            # Initialize base score from search
            base_score = result["score"]
            
            # Normalize scores based on source (KNN vs BM25)
            if result["source"] == "knn":
                # KNN scores are already normalized between 0-1
                normalized_score = base_score
            else:
                # Normalize BM25 scores to 0-1 range (assuming max BM25 score is around 10)
                normalized_score = min(base_score / 10.0, 1.0)
            
            # Add recency factor (if timestamp exists)
            recency_score = 0.0
            if isinstance(entry, dict) and "timestamp" in entry:
                # Calculate recency score (higher for more recent entries)
                # Decay over time (1 day = significant decay)
                time_diff_days = (current_time - entry["timestamp"]) / (24 * 3600)
                recency_score = 1.0 / (1.0 + time_diff_days)
            
            # Add importance factor (if importance exists)
            importance_score = 0.0
            if isinstance(entry, dict) and "importance" in entry:
                importance_score = float(entry["importance"])
            else:
                importance_score = 0.5  # Default importance
            
            # Calculate final score with weights
            # 60% relevance, 20% recency, 20% importance
            result["final_score"] = (0.6 * normalized_score) + (0.2 * recency_score) + (0.2 * importance_score)
        
        # Sort by final score (descending)
        combined_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Remove duplicates (keep highest score)
        unique_results = []
        seen_texts = set()
        for result in combined_results:
            result_text = result["text"]
            if result_text not in seen_texts:
                unique_results.append(result)
                seen_texts.add(result_text)
        
        # Format results as text
        result_texts = []
        for i, result in enumerate(unique_results[:max_results]):
            entry_text = result["text"]
            
            # Add metadata if available
            metadata = ""
            if isinstance(result["entry"], dict):
                if "importance" in result["entry"]:
                    metadata += f", importance: {result['entry']['importance']:.2f}"
                if "timestamp" in result["entry"]:
                    # Convert timestamp to readable format
                    import datetime
                    date_str = datetime.datetime.fromtimestamp(result["entry"]["timestamp"]).strftime('%Y-%m-%d %H:%M')
                    metadata += f", date: {date_str}"
            
            result_texts.append(f"{i+1}. {entry_text} (score: {result['final_score']:.2f}, source: {result['source']}{metadata})")
        
        formatted_results = "\n\n".join(result_texts)
        
        print(f"Combined search found {len(unique_results)} unique results")
        
        # If summarize_after_retrieval is enabled, summarize the combined results
        if self.summarize_after_retrieval and formatted_results:
            return self._summarize_retrieved_memories(query, formatted_results)
        
        return formatted_results
    
    def _summarize_retrieved_memories(self, query, memories):
        """Summarize retrieved memories using LLM"""
        try:
            # Prepare the prompt
            prompt = f"Summarize these memories to help answer: {query}\n\n{memories}"
            
            # Call Ollama API to summarize
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "")
                print(f"Summarized {len(memories.split())} words of memories into {len(summary.split())} words")
                return summary
            else:
                print(f"Error summarizing memories: {response.status_code}")
                return memories
        except Exception as e:
            print(f"Error summarizing memories: {e}")
            return memories

def optimize_rag_system(ollama_chat_instance=None):
    """Replace the standard RAG handler with the optimized one
    
    Args:
        ollama_chat_instance: The OllamaChat instance to optimize
        
    Returns:
        bool: True if optimization was successful, False otherwise
    """
    try:
        # If no instance provided, try to import from server
        if ollama_chat_instance is None:
            try:
                from server import ollama_chat as server_ollama_chat
                ollama_chat_instance = server_ollama_chat
            except ImportError as e:
                print(f"Error importing ollama_chat from server: {e}")
                return False
        
        # Create a new optimized RAG handler with the same config
        optimized_rag = OptimizedRAGHandler(config=ollama_chat_instance.rag.thresholds)
        
        # Copy over any existing data
        optimized_rag.entity_tracker = ollama_chat_instance.rag.entity_tracker
        
        # Replace the RAG handler
        ollama_chat_instance.rag = optimized_rag
        
        print("RAG system optimized with improvements from updates.md")
        return True
    except Exception as e:
        print(f"Error optimizing RAG system: {e}")
        return False 