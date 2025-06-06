"""
Conversation Context Manager

Maintains conversation history and context for multi-turn interactions,
enabling proper image editing and continuation workflows.
"""

import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

@dataclass
class GenerationResult:
    """Represents a single generation result in conversation history."""
    timestamp: float
    prompt: str
    intent: str
    result_type: str  # "image", "text", "video", etc.
    result_data: Dict[str, Any]  # URLs, metadata, etc.
    agent_name: str
    request_id: str
    
    def is_recent(self, minutes: int = 30) -> bool:
        """Check if this result is recent enough to be relevant."""
        age_minutes = (time.time() - self.timestamp) / 60
        return age_minutes <= minutes

@dataclass  
class ConversationTurn:
    """Represents a complete conversation turn with input and output."""
    user_input: str
    timestamp: float
    results: List[GenerationResult]
    turn_id: str

class ConversationContextManager:
    """
    Manages conversation context and history for multi-turn interactions.
    
    Enables agents to:
    - Reference previous generations for editing
    - Maintain conversation continuity
    - Detect edit vs creation intents
    - Pass appropriate context to tools
    """
    
    def __init__(self, max_history_length: int = 50, context_window_minutes: int = 60):
        """
        Initialize conversation context manager.
        
        Args:
            max_history_length: Maximum number of turns to keep
            context_window_minutes: How long results stay "active" for editing
        """
        self.conversation_turns: List[ConversationTurn] = []
        self.generation_results: List[GenerationResult] = []
        self.max_history_length = max_history_length
        self.context_window_minutes = context_window_minutes
        
        # Quick lookup for recent image results
        self._recent_images: Dict[str, GenerationResult] = {}
        
    def add_generation_result(
        self, 
        prompt: str,
        intent: str,
        result_type: str,
        result_data: Dict[str, Any],
        agent_name: str,
        request_id: str
    ) -> GenerationResult:
        """
        Add a generation result to conversation history.
        
        Args:
            prompt: Original user prompt
            intent: Detected intent (generate_image, edit_image, etc.)
            result_type: Type of result (image, text, video)
            result_data: Result data (URLs, metadata)
            agent_name: Agent that generated this result
            request_id: Request ID for tracking
            
        Returns:
            GenerationResult object added to history
        """
        result = GenerationResult(
            timestamp=time.time(),
            prompt=prompt,
            intent=intent,
            result_type=result_type,
            result_data=result_data,
            agent_name=agent_name,
            request_id=request_id
        )
        
        self.generation_results.append(result)
        
        # Update recent images lookup for quick access
        if result_type == "image" and "image_url" in result_data:
            self._recent_images[request_id] = result
        
        # Clean up old results
        self._cleanup_old_results()
        
        return result
    
    def get_recent_images(self, limit: int = 5) -> List[GenerationResult]:
        """
        Get recent image generation results that could be edited.
        
        Args:
            limit: Maximum number of images to return
            
        Returns:
            List of recent image GenerationResult objects
        """
        recent_images = []
        
        for result in reversed(self.generation_results):
            if (result.result_type == "image" and 
                result.is_recent(self.context_window_minutes) and
                len(recent_images) < limit):
                recent_images.append(result)
        
        return recent_images
    
    def get_most_recent_image(self) -> Optional[GenerationResult]:
        """
        Get the most recent image generation for editing.
        
        Returns:
            Most recent image GenerationResult or None
        """
        recent_images = self.get_recent_images(limit=1)
        return recent_images[0] if recent_images else None
    
    def detect_edit_context(self, current_prompt: str, detected_intent: str) -> Dict[str, Any]:
        """
        Analyze if current prompt is editing a previous generation.
        
        Args:
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
            "confidence": 0.0
        }
        
        # Check if intent suggests editing
        if detected_intent in ["edit_image", "modify_image", "style_transfer"]:
            context["is_edit"] = True
            context["confidence"] = 0.9
        
        # Check for edit keywords in prompt
        edit_keywords = [
            "add", "remove", "change", "modify", "edit", "alter", 
            "put", "place", "insert", "delete", "replace", "swap"
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
        
        # If editing is detected, find target image
        if context["is_edit"]:
            recent_image = self.get_most_recent_image()
            if recent_image:
                context["target_image"] = recent_image.result_data.get("image_url")
                context["original_prompt"] = recent_image.prompt
                context["confidence"] = min(context["confidence"] + 0.2, 1.0)
            else:
                # No recent image found - downgrade confidence
                context["confidence"] *= 0.5
        
        return context
    
    def get_conversation_summary(self, turns: int = 5) -> Dict[str, Any]:
        """
        Get a summary of recent conversation for context.
        
        Args:
            turns: Number of recent turns to include
            
        Returns:
            Conversation summary with key information
        """
        recent_results = self.generation_results[-turns:] if len(self.generation_results) >= turns else self.generation_results
        
        summary = {
            "total_generations": len(self.generation_results),
            "recent_generations": len(recent_results),
            "has_recent_images": len(self.get_recent_images()) > 0,
            "conversation_history": []
        }
        
        for result in recent_results:
            summary["conversation_history"].append({
                "prompt": result.prompt[:100] + "..." if len(result.prompt) > 100 else result.prompt,
                "intent": result.intent,
                "result_type": result.result_type,
                "timestamp": result.timestamp,
                "has_image": "image_url" in result.result_data
            })
        
        return summary
    
    def prepare_edit_context_for_tools(self, edit_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context data for tool consumption.
        
        Args:
            edit_context: Edit context from detect_edit_context()
            
        Returns:
            Tool-ready context data
        """
        if not edit_context["is_edit"]:
            return {}
        
        tool_context = {
            "mode": "edit",
            "edit_type": edit_context["edit_type"],
            "edit_instructions": edit_context["edit_instructions"],
            "confidence": edit_context["confidence"]
        }
        
        if edit_context["target_image"]:
            tool_context["original_image"] = edit_context["target_image"]
        
        if edit_context["original_prompt"]:
            tool_context["original_prompt"] = edit_context["original_prompt"]
            tool_context["combined_prompt"] = f"{edit_context['original_prompt']} + {edit_context['edit_instructions']}"
        
        return tool_context
    
    def _cleanup_old_results(self):
        """Clean up old results to maintain memory limits."""
        # Remove results older than context window
        cutoff_time = time.time() - (self.context_window_minutes * 60)
        self.generation_results = [
            result for result in self.generation_results 
            if result.timestamp > cutoff_time
        ]
        
        # Enforce max history length
        if len(self.generation_results) > self.max_history_length:
            self.generation_results = self.generation_results[-self.max_history_length:]
        
        # Update recent images lookup
        self._recent_images = {
            rid: result for rid, result in self._recent_images.items()
            if result.timestamp > cutoff_time
        }
    
    def export_context(self) -> Dict[str, Any]:
        """Export conversation context for debugging/analysis."""
        return {
            "total_results": len(self.generation_results),
            "recent_images": len(self.get_recent_images()),
            "context_window_minutes": self.context_window_minutes,
            "results": [asdict(result) for result in self.generation_results]
        }

# Global conversation context manager
conversation_context = ConversationContextManager() 