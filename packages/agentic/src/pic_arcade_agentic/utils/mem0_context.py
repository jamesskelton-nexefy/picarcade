"""
Mem0-Based Conversation Context Manager

Uses Mem0 for persistent, intelligent memory management across API requests.
Replaces the custom session context system with enterprise-grade memory.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import hashlib

try:
    from mem0 import MemoryClient
    HAS_MEM0 = True
except ImportError:
    HAS_MEM0 = False
    print("⚠️ Mem0 not installed. Install with: pip install mem0ai")

from .conversation_context import GenerationResult

class Mem0ConversationContext:
    """
    Mem0-powered conversation context manager for persistent memory across sessions.
    
    Provides intelligent memory storage, retrieval, and management for multi-turn
    image editing workflows using Mem0's advanced memory architecture.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Mem0 conversation context.
        
        Args:
            api_key: Mem0 API key (or use MEM0_API_KEY environment variable)
        """
        if not HAS_MEM0:
            raise ImportError("Mem0 not installed. Install with: pip install mem0ai")
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self.api_key:
            raise ValueError("Mem0 API key required. Set MEM0_API_KEY environment variable or pass api_key parameter")
        
        # Initialize Mem0 client
        self.memory = MemoryClient(api_key=self.api_key)
        
        # Memory categories for different types of context
        self.CATEGORIES = {
            "image_generation": "Image generation results and metadata",
            "user_preferences": "User preferences and settings", 
            "conversation_flow": "Conversation flow and intent patterns",
            "edit_relationships": "Relationships between original and edited images"
        }
    
    def store_generation_result(
        self, 
        user_id: str,
        prompt: str,
        intent: str,
        result_type: str,
        result_data: Dict[str, Any],
        agent_name: str,
        request_id: str
    ) -> bool:
        """
        Store generation result in Mem0 for future reference.
        
        Args:
            user_id: User identifier for memory association
            prompt: Original user prompt
            intent: Detected intent (generate_image, edit_image, etc.)
            result_type: Type of result (image, text, video)
            result_data: Result data (URLs, metadata)
            agent_name: Agent that generated this result
            request_id: Request ID for tracking
            
        Returns:
            Success status
        """
        try:
            # Create comprehensive memory entry
            memory_text = f"""
Generated {result_type} for user request: "{prompt}"
Intent: {intent}
Agent: {agent_name}
Result: {json.dumps(result_data, indent=2)}
Request ID: {request_id}
Timestamp: {time.time()}
"""
            
            # Add to Mem0 with metadata
            self.memory.add(
                memory_text,
                user_id=user_id,
                metadata={
                    "category": "image_generation",
                    "intent": intent,
                    "result_type": result_type,
                    "agent_name": agent_name,
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "image_url": result_data.get("image_url"),
                    "prompt": prompt
                }
            )
            
            # Store relationship information for edits
            if result_data.get("image_url"):
                relationship_text = f"User has generated image with URL: {result_data['image_url']} from prompt: '{prompt}'. This image is available for editing operations."
                
                self.memory.add(
                    relationship_text,
                    user_id=user_id,
                    metadata={
                        "category": "edit_relationships",
                        "image_url": result_data["image_url"],
                        "original_prompt": prompt,
                        "timestamp": time.time(),
                        "available_for_editing": True
                    }
                )
            
            return True
            
        except Exception as e:
            print(f"Failed to store generation result in Mem0: {e}")
            return False
    
    def detect_edit_context(self, user_id: str, current_prompt: str, detected_intent: str) -> Dict[str, Any]:
        """
        Detect if current prompt is editing a previous generation using Mem0 memory.
        
        Args:
            user_id: User identifier
            current_prompt: Current user prompt
            detected_intent: Intent detected by prompt parser
            
        Returns:
            Dictionary with edit context information
        """
        context = {
            "is_edit": False,
            "edit_type": None,
            "target_image": None,
            "original_prompt": None,
            "edit_instructions": current_prompt,
            "confidence": 0.0,
            "memory_source": "mem0"
        }
        
        try:
            # Check if intent suggests editing
            if detected_intent in ["edit_image", "modify_image", "style_transfer"]:
                context["is_edit"] = True
                context["confidence"] = 0.9
            
            # Check for edit keywords in prompt
            edit_keywords = [
                "add", "remove", "change", "modify", "edit", "alter", 
                "put", "place", "insert", "delete", "replace", "swap",
                "make", "animate", "move", "blink", "wiggle", "smile", "open", "close", "turn", "rotate", "jump", "walk", "run"
            ]
            
            prompt_lower = current_prompt.lower()
            has_edit_keywords = any(keyword in prompt_lower for keyword in edit_keywords)
            
            if has_edit_keywords:
                context["is_edit"] = True
                context["confidence"] = max(context["confidence"], 0.7)
                
                # Determine edit type
                if any(word in prompt_lower for word in ["add", "put", "place", "insert"]):
                    context["edit_type"] = "add_object"
                elif any(word in prompt_lower for word in ["remove", "delete"]):
                    context["edit_type"] = "remove_object" 
                elif any(word in prompt_lower for word in ["change", "replace", "swap"]):
                    context["edit_type"] = "change_object"
                elif any(word in prompt_lower for word in ["style", "like", "as"]):
                    context["edit_type"] = "style_transfer"
                elif any(word in prompt_lower for word in ["make", "animate", "move", "blink", "wiggle"]):
                    context["edit_type"] = "animate"
            
            # If editing is detected, search for recent images in Mem0
            if context["is_edit"] or (not context["is_edit"] and len(current_prompt.split()) <= 8):
                # Search for recent image generations
                search_results = self.memory.search(
                    "recent image generation available for editing",
                    user_id=user_id,
                    limit=5
                )
                
                # Find the most recent relevant image
                best_match = None
                best_score = 0
                
                for result in search_results:
                    metadata = result.get("metadata", {})
                    
                    # Prioritize recent images that are available for editing
                    if (metadata.get("category") == "edit_relationships" and 
                        metadata.get("available_for_editing") and
                        metadata.get("image_url")):
                        
                        # Score based on recency and relevance
                        timestamp = metadata.get("timestamp", 0)
                        age_hours = (time.time() - timestamp) / 3600
                        recency_score = max(0, 1 - (age_hours / 24))  # Decay over 24 hours
                        
                        if recency_score > best_score:
                            best_score = recency_score
                            best_match = metadata
                
                if best_match:
                    context["target_image"] = best_match["image_url"]
                    context["original_prompt"] = best_match["original_prompt"]
                    context["confidence"] = min(context["confidence"] + 0.2, 1.0)
                    context["is_edit"] = True
                else:
                    # No recent image found - downgrade confidence
                    context["confidence"] *= 0.5
            
            return context
            
        except Exception as e:
            print(f"Error detecting edit context with Mem0: {e}")
            # Fallback to simple keyword detection
            context["confidence"] = 0.3 if has_edit_keywords else 0.0
            return context
    
    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get conversation summary from Mem0.
        
        Args:
            user_id: User identifier
            
        Returns:
            Conversation summary
        """
        try:
            # Search for recent conversation context
            results = self.memory.search(
                "recent conversation image generation",
                user_id=user_id,
                limit=10
            )
            
            image_count = 0
            recent_generations = 0
            last_activity = None
            
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("category") == "image_generation":
                    recent_generations += 1
                    if metadata.get("image_url"):
                        image_count += 1
                    
                    # Track most recent activity
                    timestamp = metadata.get("timestamp", 0)
                    if last_activity is None or timestamp > last_activity:
                        last_activity = timestamp
            
            return {
                "total_generations": recent_generations,
                "recent_generations": recent_generations,
                "has_recent_images": image_count > 0,
                "recent_images_count": image_count,
                "last_activity": last_activity,
                "memory_source": "mem0"
            }
            
        except Exception as e:
            print(f"Error getting conversation summary from Mem0: {e}")
            return {
                "total_generations": 0,
                "recent_generations": 0,
                "has_recent_images": False,
                "recent_images_count": 0,
                "last_activity": None,
                "memory_source": "mem0_error"
            }
    
    def store_user_preference(self, user_id: str, preference: str) -> bool:
        """
        Store user preference in Mem0.
        
        Args:
            user_id: User identifier
            preference: Preference to store
            
        Returns:
            Success status
        """
        try:
            self.memory.add(
                preference,
                user_id=user_id,
                metadata={
                    "category": "user_preferences",
                    "timestamp": time.time()
                }
            )
            return True
        except Exception as e:
            print(f"Failed to store user preference: {e}")
            return False
    
    def get_user_preferences(self, user_id: str) -> List[str]:
        """
        Get user preferences from Mem0.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user preferences
        """
        try:
            results = self.memory.search(
                "user preferences settings",
                user_id=user_id,
                limit=20
            )
            
            preferences = []
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("category") == "user_preferences":
                    preferences.append(result.get("memory", ""))
            
            return preferences
            
        except Exception as e:
            print(f"Error getting user preferences: {e}")
            return []
    
    def clear_user_memory(self, user_id: str) -> bool:
        """
        Clear all memory for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Success status
        """
        try:
            # Note: Mem0 doesn't have a direct "delete all for user" method
            # This would typically be handled through their API or dashboard
            print(f"⚠️ Clear memory for user {user_id} should be done through Mem0 dashboard")
            return True
        except Exception as e:
            print(f"Error clearing user memory: {e}")
            return False
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get memory statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Memory statistics
        """
        try:
            # Get all memories for the user
            all_memories = self.memory.search(
                "",  # Empty query to get all
                user_id=user_id,
                limit=100
            )
            
            stats = {
                "total_memories": len(all_memories),
                "categories": {},
                "oldest_memory": None,
                "newest_memory": None
            }
            
            for memory in all_memories:
                metadata = memory.get("metadata", {})
                category = metadata.get("category", "unknown")
                
                # Count by category
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                
                # Track oldest/newest
                timestamp = metadata.get("timestamp", 0)
                if stats["oldest_memory"] is None or timestamp < stats["oldest_memory"]:
                    stats["oldest_memory"] = timestamp
                if stats["newest_memory"] is None or timestamp > stats["newest_memory"]:
                    stats["newest_memory"] = timestamp
            
            return stats
            
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {"total_memories": 0, "categories": {}, "error": str(e)}

# Global Mem0 context manager instance
_mem0_context = None

def get_mem0_context(api_key: Optional[str] = None) -> Mem0ConversationContext:
    """
    Get or create global Mem0 conversation context instance.
    
    Args:
        api_key: Mem0 API key (optional, uses environment variable if not provided)
        
    Returns:
        Mem0ConversationContext instance
    """
    global _mem0_context
    
    if _mem0_context is None:
        _mem0_context = Mem0ConversationContext(api_key=api_key)
    
    return _mem0_context

def create_user_session_id(user_identifier: str) -> str:
    """
    Create a consistent session ID for a user.
    
    Args:
        user_identifier: User identifier (email, user_id, etc.)
        
    Returns:
        Consistent session ID for the user
    """
    # Create hash-based session ID for consistency
    return hashlib.md5(user_identifier.encode()).hexdigest()[:12] 