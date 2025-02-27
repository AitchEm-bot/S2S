import chromadb
from sentence_transformers import SentenceTransformer
import json

class RAGHandler:
    def __init__(self):
        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        # Create or get existing collection
        self.collection = self.chroma_client.get_or_create_collection("chat_context")
        # Load the embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def store_interaction(self, message):
        """Store only user message in vector DB"""
        print(f"Storing in RAG - Message: {message}")
        
        # Store only user message
        embedding = self.embed_model.encode(message).tolist()
        
        self.collection.add(
            ids=[str(hash(message))],
            embeddings=[embedding],
            metadatas=[{"text": message}]
        )
        print("Successfully stored in RAG")

    def get_relevant_context(self, query, k=3):
        """Retrieve relevant context for the query"""
        print(f"Searching RAG for: {query}")
        
        query_embedding = self.embed_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        print(f"Found {len(results['metadatas'][0])} relevant contexts")
        for doc in results["metadatas"][0]:
            print(f"Retrieved: {doc['text']}")
        
        return [doc["text"] for doc in results["metadatas"][0]]

    def format_context(self, relevant_docs):
        """Format retrieved context for the LLM"""
        if not relevant_docs:
            return ""
        
        context = "Here's some relevant context from previous conversations:\n"
        context += "\n---\n".join(relevant_docs)
        context += "\nPlease use this context to inform your response if relevant."
        return context

    def print_all_stored(self):
        """Debug method to print all stored interactions"""
        try:
            all_results = self.collection.get()
            print("\nAll stored interactions:")
            for i, text in enumerate(all_results['metadatas']):
                print(f"{i+1}. {text['text']}\n")
        except Exception as e:
            print(f"Error retrieving stored interactions: {e}")

    def clear_collection(self):
        """Clear all stored data"""
        try:
            print("Starting RAG clear process...")
            
            # Get all document IDs first
            all_results = self.collection.get()
            if 'ids' in all_results and all_results['ids']:
                # Delete all documents using their IDs
                self.collection.delete(ids=all_results['ids'])
                print(f"Deleted {len(all_results['ids'])} documents")
            else:
                print("No documents to delete")
            
            # Verify collection is empty
            results = self.collection.get()
            count = len(results.get('ids', []))
            print(f"Collection now has {count} items")
            
            return True
        except Exception as e:
            print(f"Error clearing RAG memory: {e}")
            return False