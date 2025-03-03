import spacy
import time
from collections import defaultdict, Counter

class EntityTracker:
    """
    Tracks entities mentioned in conversations and determines their importance.
    """
    
    def __init__(self):
        """Initialize the entity tracker with spaCy NLP model."""
        self.nlp = spacy.load("en_core_web_sm")
        
        # Store entities with their metadata
        self.entities = defaultdict(lambda: {
            "mentions": 0,
            "first_seen": time.time(),
            "last_seen": time.time(),
            "contexts": [],
            "types": Counter(),
            "importance": 0.0
        })
        
        # Define entity types of interest with their importance weights
        self.entity_weights = {
            "PERSON": 1.0,      # Names of people
            "ORG": 0.8,         # Organizations
            "GPE": 0.8,         # Geopolitical entities (countries, cities)
            "LOC": 0.7,         # Non-GPE locations
            "DATE": 0.7,        # Dates
            "TIME": 0.6,        # Times
            "EVENT": 0.9,       # Events
            "WORK_OF_ART": 0.6, # Titles of books, songs, etc.
            "FAC": 0.6,         # Facilities
            "PRODUCT": 0.6      # Products
        }
        
        # Context keywords that increase importance
        self.importance_contexts = {
            "remember": 1.5,
            "important": 1.5,
            "forget": 1.5,
            "remind": 1.5,
            "note": 1.2,
            "significant": 1.3,
            "key": 1.2,
            "critical": 1.3,
            "essential": 1.2,
            "vital": 1.3,
            "crucial": 1.3,
            "major": 1.1,
            "primary": 1.1,
            "central": 1.1,
            "core": 1.1,
            "fundamental": 1.2,
            "meeting": 1.2,
            "appointment": 1.2,
            "schedule": 1.2,
            "deadline": 1.3,
            "project": 1.1,
            "task": 1.1,
            "goal": 1.1,
            "plan": 1.1,
            "future": 1.1,
            "tomorrow": 1.2,
            "next week": 1.2,
            "next month": 1.2
        }
        
        # Maximum number of contexts to store per entity
        self.max_contexts = 5
        
        # Recent messages for tracking repeated mentions
        self.recent_messages = []
        self.max_recent_messages = 10
    
    def extract_entities(self, text):
        """
        Extract named entities from text using spaCy.
        
        Args:
            text (str): The text to extract entities from
            
        Returns:
            list: List of entity dictionaries with 'text' and 'type' keys
        """
        if not text or not isinstance(text, str):
            return []
            
        doc = self.nlp(text)
        return [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
    
    def update_entities(self, text):
        """
        Update entity tracking based on new text.
        
        Args:
            text (str): The text to process
            
        Returns:
            list: List of important entities found in this text
        """
        if not text or not isinstance(text, str):
            return []
            
        # Add to recent messages
        self.recent_messages.append(text)
        if len(self.recent_messages) > self.max_recent_messages:
            self.recent_messages.pop(0)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Track important entities found in this text
        important_entities = []
        
        # Update entity records
        for entity in entities:
            entity_text = entity["text"]
            entity_type = entity["type"]
            
            # Normalize entity text
            entity_key = entity_text.lower()
            
            # Update entity record
            entity_data = self.entities[entity_key]
            entity_data["mentions"] += 1
            entity_data["last_seen"] = time.time()
            entity_data["types"][entity_type] += 1
            
            # Add context (truncated text around entity)
            context = self._get_context(text, entity_text)
            if context and len(entity_data["contexts"]) < self.max_contexts:
                entity_data["contexts"].append(context)
            
            # Calculate importance score
            importance = self._calculate_entity_importance(entity_key, entity_type, text)
            entity_data["importance"] = importance
            
            # If important enough, add to the list of important entities
            if importance > 0.6:
                important_entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "importance": importance
                })
        
        return important_entities
    
    def _get_context(self, text, entity_text):
        """
        Extract a window of text around the entity mention.
        
        Args:
            text (str): The full text
            entity_text (str): The entity text to find
            
        Returns:
            str: Context window around the entity
        """
        # Find the entity in the text
        entity_pos = text.lower().find(entity_text.lower())
        if entity_pos == -1:
            return ""
        
        # Extract context (50 chars before and after)
        start = max(0, entity_pos - 50)
        end = min(len(text), entity_pos + len(entity_text) + 50)
        
        return text[start:end]
    
    def _calculate_entity_importance(self, entity_key, entity_type, context_text):
        """
        Calculate importance score for an entity based on various factors.
        
        Args:
            entity_key (str): The normalized entity text
            entity_type (str): The entity type
            context_text (str): The text containing the entity
            
        Returns:
            float: Importance score between 0 and 1
        """
        entity_data = self.entities[entity_key]
        score = 0.0
        
        # Factor 1: Entity type importance
        type_score = self.entity_weights.get(entity_type, 0.3)
        score += type_score * 0.3
        
        # Factor 2: Mention frequency - Increased weight for repeated mentions
        # Increase the cap to 10 mentions and the weight to 0.3
        mention_score = min(1.0, entity_data["mentions"] / 10)  # Cap at 10 mentions
        score += mention_score * 0.3  # Increased from 0.2 to 0.3
        
        # Factor 3: Recency (higher if mentioned recently)
        recency = 1.0  # Default high if it's a new entity
        if len(self.recent_messages) > 1:
            # Check if mentioned in previous messages
            prev_mentions = sum(1 for msg in self.recent_messages[:-1] if entity_key in msg.lower())
            if prev_mentions > 0:
                recency = min(1.0, prev_mentions / 3)  # Cap at 3 previous mentions
        score += recency * 0.2
        
        # Factor 4: Context keywords - Increased weight for "remember" and similar keywords
        context_score = 0.0
        context_lower = context_text.lower()
        
        # Special boost for "remember" context
        if "remember" in context_lower:
            context_score = 0.8  # Strong boost for explicit remember requests
        else:
            for keyword, weight in self.importance_contexts.items():
                if keyword in context_lower:
                    context_score = max(context_score, weight - 1.0)  # Convert weight to a 0-1 scale
        
        # Increased weight for context keywords
        score += context_score * 0.4  # Increased from 0.3 to 0.4
        
        # Bonus for PERSON entities that are mentioned multiple times
        if entity_type == "PERSON" and entity_data["mentions"] >= 3:
            score += 0.2  # Additional boost for frequently mentioned people
        
        # Ensure score is between 0 and 1
        return min(1.0, max(0.0, score))
    
    def get_entity_summary(self, entity_key):
        """
        Get a summary of information about an entity.
        
        Args:
            entity_key (str): The normalized entity text
            
        Returns:
            dict: Entity information or None if not found
        """
        if entity_key.lower() not in self.entities:
            return None
            
        entity_data = self.entities[entity_key.lower()]
        
        # Get the most common entity type
        most_common_type = max(entity_data["types"].items(), key=lambda x: x[1])[0] if entity_data["types"] else "UNKNOWN"
        
        return {
            "text": entity_key,
            "type": most_common_type,
            "mentions": entity_data["mentions"],
            "first_seen": entity_data["first_seen"],
            "last_seen": entity_data["last_seen"],
            "contexts": entity_data["contexts"],
            "importance": entity_data["importance"]
        }
    
    def get_important_entities(self, min_importance=0.6):
        """
        Get all entities above a certain importance threshold.
        
        Args:
            min_importance (float): Minimum importance score (0-1)
            
        Returns:
            list: List of important entities with their data
        """
        important = []
        
        for entity_key, entity_data in self.entities.items():
            if entity_data["importance"] >= min_importance:
                # Get the most common entity type
                most_common_type = max(entity_data["types"].items(), key=lambda x: x[1])[0] if entity_data["types"] else "UNKNOWN"
                
                important.append({
                    "text": entity_key,
                    "type": most_common_type,
                    "mentions": entity_data["mentions"],
                    "importance": entity_data["importance"],
                    "contexts": entity_data["contexts"]
                })
        
        # Sort by importance (highest first)
        important.sort(key=lambda x: x["importance"], reverse=True)
        
        return important
    
    def find_entities_in_query(self, query):
        """
        Find entities in a query and return relevant stored entities.
        
        Args:
            query (str): The query text
            
        Returns:
            list: List of relevant entities with their data
        """
        if not query or not isinstance(query, str):
            return []
            
        # Extract entities from the query
        query_entities = self.extract_entities(query)
        
        # Find matching stored entities
        matches = []
        
        for entity in query_entities:
            entity_text = entity["text"]
            entity_type = entity["type"]
            
            entity_key = entity_text.lower()
            
            if entity_key in self.entities:
                entity_data = self.get_entity_summary(entity_key)
                if entity_data:
                    matches.append(entity_data)
        
        return matches 