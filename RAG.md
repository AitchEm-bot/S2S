# **Hybrid RAG: Optimized Storage & Retrieval**

## **📌 Overview**
This document outlines an efficient storage and retrieval strategy for a **Hybrid RAG (Retrieval-Augmented Generation)** implementation. The goal is to:
- Store **only key insights** instead of entire conversations.
- Retrieve relevant information **only when necessary** rather than injecting retrieved data into every response.
- Optimize storage with **smart filtering**, **adaptive saving**, and **hybrid memory indexing**.

---

## **1️⃣ Selective Memory Storage**
### **🔹 Key Information Extraction**
Instead of storing everything, extract only important insights:
```python
from ollama import chat

def extract_key_points(conversation_history):
    prompt = f"""
    Given the following conversation:

    {conversation_history}

    Identify the key details that are useful for future reference.
    Exclude small talk, repetitive information, and non-essential details.
    Return a concise bullet-point list of critical insights.
    """
    
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
```
✔️ Extracts meaningful details
✔️ Filters out unnecessary chatter
✔️ Ensures concise memory storage

---

## **2️⃣ Smart Filtering Before Storage**
Before storing new data, check if it is **unique and important**:

### **🔹 Duplicate Prevention & Thresholding**
```python
def should_store(new_info, existing_memory):
    similarity_scores = [semantic_similarity(new_info, memory) for memory in existing_memory]
    
    if max(similarity_scores, default=0) > 0.8:  # High similarity → discard
        return False  

    return importance_score(new_info) > 0.7  # Store if above threshold
```
✔️ Avoids redundant storage
✔️ Stores only insights above a certain importance level

---

## **3️⃣ Adaptive Storage Mechanism**
To prevent unnecessary storage, save only **at intervals or when needed**:
```python
import time

storage_buffer = []

def store_memory(new_info):
    global storage_buffer
    storage_buffer.append(new_info)

    # Store every 5 minutes or when buffer exceeds a threshold
    if len(storage_buffer) >= 5 or time.time() % 300 == 0:
        persistent_store.extend(storage_buffer)
        storage_buffer = []  # Clear buffer
```
✔️ Prevents excessive storage calls
✔️ Saves data efficiently

---

## **4️⃣ Hybrid Memory Indexing**
Instead of **storing everything in one place**, organize memory into:
- **Short-term memory** (session-specific context)
- **Long-term memory** (persistent knowledge base)
- **Ephemeral memory** (temporary, erased after session)

### **🔹 Categorizing Storage**
```python
def store_in_memory(new_info):
    if is_short_term(new_info):
        short_term_memory.append(new_info)
    elif is_long_term(new_info):
        vector_database.store(new_info)
    else:
        ephemeral_memory.append(new_info)  # Session-only data
```
✔️ Keeps memory modular
✔️ Short-term for temporary facts
✔️ Long-term for user preferences
✔️ Ephemeral for session-only data

---

## **5️⃣ Optimized Retrieval**
### **🔹 Only Retrieve When Needed**
Instead of injecting retrieved data **into every response**, use a **conditional retrieval mechanism**:
```python
def retrieve_relevant_data(query, context):
    if should_retrieve(query, context):
        return vector_database.search(query)
    return ""

def should_retrieve(query, context):
    return importance_score(query) > 0.6  # Retrieve only if query is significant
```
✔️ Prevents unnecessary retrieval
✔️ Only searches for information when relevant

---

## **🎯 Final Takeaways**
✅ **Selective Storage** – Extract only key points
✅ **Smart Filtering** – Prevent redundant or unimportant storage
✅ **Adaptive Saving** – Store data only when necessary
✅ **Hybrid Indexing** – Organize memory for efficient retrieval
✅ **Optimized Retrieval** – Retrieve information only when relevant

## **🔍 Implementation Notes**
### **Current Implementation**
- Using ChromaDB for vector storage
- SentenceTransformer for embedding generation
- Top-k retrieval (k=3) for context
- System message injection with retrieved context

### **Observed Behaviors**
- LLM tends to reference retrieved context heavily
- Context is retrieved for every message
- Persistence between server restarts via ChromaDB
- Reset functionality clears both immediate and stored context

### **Potential Optimizations**
- Adjust similarity thresholds
- Implement conditional retrieval
- Fine-tune context prompt formatting
- Add importance scoring for storage decisions

By implementing these strategies, we ensure a **subtle, efficient, and scalable Hybrid RAG** system. 🚀
