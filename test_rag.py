#!/usr/bin/env python3
"""
Test script for the enhanced RAG system.
This script runs a series of tests to verify the functionality of the RAG system.
"""

import unittest
import time
from rag_handler import RAGHandler

class TestRAGSystem(unittest.TestCase):
    """Test cases for the RAG system"""
    
    def setUp(self):
        """Set up the test environment"""
        # Initialize RAG with test configuration
        self.rag = RAGHandler(config={
            "storage_min": 0.3,
            "ephemeral_max": 0.4,
            "short_term_max": 0.7,
            "retrieval_min": 0.3,
            "similarity_max": 0.8,
            "ephemeral_similarity": 0.5,
            "key_points_min_length": 30,
            "short_term_expiry_days": 0.001  # Set to a very small value for testing (about 1.5 minutes)
        })
        
        # Clear any existing data
        self.rag.clear_collection()
        
    def tearDown(self):
        """Clean up after tests"""
        self.rag.clear_collection()
        
    def test_memory_categorization(self):
        """Test memory categorization based on importance"""
        # High importance message (should go to long-term memory)
        high_importance = "This is a critical security vulnerability in our system that needs immediate attention. The authentication module has a bypass that allows unauthorized access."
        
        # Medium importance message (should go to short-term memory)
        medium_importance = "Remember to update the documentation for the new feature we discussed yesterday."
        
        # Low importance message (should go to ephemeral memory)
        low_importance = "I think I'll have a sandwich for lunch today."
        
        # Store messages
        self.rag.store_interaction(high_importance, "I'll address this security issue right away.")
        self.rag.store_interaction(medium_importance, "I'll update the documentation.")
        self.rag.store_interaction(low_importance, "Enjoy your lunch!")
        
        # Flush buffer to ensure storage
        self.rag._flush_buffer()
        
        # Check long-term memory
        long_term_results = self.rag.long_term_memory.get()
        self.assertTrue(any("security vulnerability" in doc for doc in long_term_results.get('documents', [])), 
                       "High importance message not found in long-term memory")
        
        # Check short-term memory
        short_term_results = self.rag.short_term_memory.get()
        self.assertTrue(any("documentation" in doc for doc in short_term_results.get('documents', [])), 
                       "Medium importance message not found in short-term memory")
        
        # Check ephemeral memory
        self.assertTrue(any("sandwich" in item.get('text', '') for item in self.rag.ephemeral_memory), 
                       "Low importance message not found in ephemeral memory")
        
    def test_key_point_extraction(self):
        """Test key point extraction from long text"""
        long_text = """
        This is a very long message with lots of details. The main point is that we need to refactor the database layer.
        There are several issues with the current implementation:
        1. Connection pooling is not optimized
        2. Queries are not parameterized properly
        3. There's no proper error handling
        4. The ORM mapping is inefficient
        
        We should prioritize fixing the connection pooling and parameterized queries first, then address the other issues.
        Also, I had cereal for breakfast this morning and the weather is quite nice today, but that's not important.
        """
        
        key_points = self.rag.extract_key_points(long_text)
        
        # Check that key points contain important information
        self.assertIn("refactor", key_points.lower(), "Key point about refactoring not extracted")
        self.assertIn("database", key_points.lower(), "Key point about database not extracted")
        
        # Check that unimportant details are excluded
        self.assertNotIn("cereal", key_points.lower(), "Unimportant detail about breakfast was not filtered out")
        self.assertNotIn("weather", key_points.lower(), "Unimportant detail about weather was not filtered out")
        
    def test_duplicate_detection(self):
        """Test duplicate detection in storage"""
        message = "We need to schedule a meeting to discuss the project timeline."
        
        # Store the message twice
        self.rag.store_interaction(message, "I'll set up the meeting.")
        self.rag._flush_buffer()
        
        # Try to store it again
        self.rag.store_interaction(message, "Let's schedule it for tomorrow.")
        self.rag._flush_buffer()
        
        # Check that it's only stored once
        all_results = self.rag.long_term_memory.get()
        matching_docs = [doc for doc in all_results.get('documents', []) if "schedule a meeting" in doc]
        self.assertEqual(len(matching_docs), 1, "Duplicate message was stored")
        
    def test_memory_expiration(self):
        """Test that short-term memories expire"""
        # Store a message in short-term memory
        medium_importance = "Please review the pull request #123 when you have time."
        self.rag.store_interaction(medium_importance, "I'll review it soon.")
        self.rag._flush_buffer()
        
        # Verify it's in short-term memory
        short_term_before = self.rag.short_term_memory.get()
        self.assertTrue(any("pull request" in doc for doc in short_term_before.get('documents', [])), 
                       "Message not found in short-term memory")
        
        # Wait for expiration (using a very short expiry time for testing)
        print("Waiting for short-term memory expiration...")
        time.sleep(90)  # 1.5 minutes
        
        # Run maintenance to trigger expiration
        self.rag.maintain_memory()
        
        # Check that it's been removed
        short_term_after = self.rag.short_term_memory.get()
        self.assertFalse(any("pull request" in doc for doc in short_term_after.get('documents', [])), 
                        "Expired message still found in short-term memory")
        
    def test_memory_consolidation(self):
        """Test memory consolidation (promotion from short-term to long-term)"""
        # Store a message in short-term memory
        medium_importance = "The new feature should be implemented using the strategy pattern."
        self.rag.store_interaction(medium_importance, "That's a good approach.")
        self.rag._flush_buffer()
        
        # Verify it's in short-term memory
        short_term_results = self.rag.short_term_memory.get()
        self.assertTrue(any("strategy pattern" in doc for doc in short_term_results.get('documents', [])), 
                       "Message not found in short-term memory")
        
        # Simulate multiple accesses to increase access count
        for i in range(3):
            # Get the memory ID
            memory_id = short_term_results['ids'][0]
            
            # Update metadata to simulate access
            metadata = short_term_results['metadatas'][0].copy()
            metadata['access_count'] = i + 1
            
            # Update the memory
            self.rag.short_term_memory.update(
                ids=[memory_id],
                metadatas=[metadata]
            )
        
        # Run consolidation
        self.rag.consolidate_memories()
        
        # Check that it's been promoted to long-term memory
        long_term_results = self.rag.long_term_memory.get()
        self.assertTrue(any("strategy pattern" in doc for doc in long_term_results.get('documents', [])), 
                       "Message not promoted to long-term memory")
        
        # Check that it's been removed from short-term memory
        short_term_after = self.rag.short_term_memory.get()
        self.assertFalse(any("strategy pattern" in doc for doc in short_term_after.get('documents', [])), 
                        "Promoted message still found in short-term memory")
        
    def test_memory_tagging(self):
        """Test automatic memory tagging"""
        # Test question tagging
        question = "How do we implement the new authentication system?"
        self.rag.store_interaction(question, "We should use OAuth 2.0 with JWT tokens.")
        
        # Test code tagging
        code = "```python\ndef calculate_total(items):\n    return sum(item.price for item in items)\n```"
        self.rag.store_interaction(code, "This function calculates the total price of items.")
        
        # Test how-to tagging
        howto = "Here are the steps to deploy the application: 1. Build the Docker image, 2. Push to registry, 3. Update Kubernetes manifests, 4. Apply the changes."
        self.rag.store_interaction(howto, "I'll follow these steps for deployment.")
        
        # Flush buffer
        self.rag._flush_buffer()
        
        # Get all tags
        all_tags = self.rag.get_all_tags()
        
        # Check that the expected tags are present
        self.assertIn("question", all_tags, "Question tag not found")
        self.assertIn("code", all_tags, "Code tag not found")
        self.assertIn("how-to", all_tags, "How-to tag not found")
        
        # Test retrieving memories by tag
        question_memories = self.rag.get_memories_by_tag("question")
        self.assertTrue(any("authentication" in memory.get('text', '') for memory in question_memories), 
                       "Question not found in memories with question tag")
        
    def test_user_feedback(self):
        """Test user feedback processing"""
        # Store a message
        message = "The database schema needs to be updated to include the new fields."
        self.rag.store_interaction(message, "I'll update the schema.")
        self.rag._flush_buffer()
        
        # Get the memory ID
        long_term_results = self.rag.long_term_memory.get()
        memory_id = long_term_results['ids'][0]
        
        # Provide positive feedback
        self.rag.process_user_feedback(memory_id, "useful", "long_term")
        
        # Check that importance was increased
        updated_results = self.rag.long_term_memory.get(ids=[memory_id])
        self.assertGreater(updated_results['metadatas'][0].get('importance', 0), 0.5, 
                          "Importance not increased after positive feedback")
        
        # Provide negative feedback
        self.rag.process_user_feedback(memory_id, "not_useful", "long_term")
        
        # Check that importance was decreased
        updated_results = self.rag.long_term_memory.get(ids=[memory_id])
        self.assertLessEqual(updated_results['metadatas'][0].get('importance', 1), 0.5, 
                           "Importance not decreased after negative feedback")

if __name__ == "__main__":
    unittest.main() 