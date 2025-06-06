"""
Session-Based Conversation Context Manager

Maintains conversation context across multiple API requests by persisting
context data and associating it with user sessions.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import threading

from .conversation_context import ConversationContextManager, GenerationResult

class SessionContextManager:
    """
    Manages conversation context across API requests using session persistence.
    
    Stores conversation context per session/user and persists to disk to maintain
    context between API calls for multi-turn editing workflows.
    """
    
    def __init__(self, storage_dir: str = "conversation_sessions"):
        """
        Initialize session context manager.
        
        Args:
            storage_dir: Directory to store session context files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # In-memory cache of active sessions
        self._session_cache: Dict[str, ConversationContextManager] = {}
        self._cache_lock = threading.Lock()
        
        # Session expiry (24 hours default)
        self.session_expiry_hours = 24
    
    def get_session_context(self, session_id: str) -> ConversationContextManager:
        """
        Get conversation context for a specific session.
        
        Args:
            session_id: Unique session identifier (user ID, session token, etc.)
            
        Returns:
            ConversationContextManager for the session
        """
        with self._cache_lock:
            # Check if session is already loaded in cache
            if session_id in self._session_cache:
                return self._session_cache[session_id]
            
            # Try to load from storage
            context = self._load_session_context(session_id)
            
            # Cache the loaded context
            self._session_cache[session_id] = context
            
            return context
    
    def save_session_context(self, session_id: str, context: ConversationContextManager) -> None:
        """
        Save conversation context for a session to persistent storage.
        
        Args:
            session_id: Session identifier
            context: Conversation context to save
        """
        try:
            session_file = self.storage_dir / f"session_{session_id}.json"
            
            # Export context data
            context_data = {
                "session_id": session_id,
                "last_updated": time.time(),
                "generation_results": [asdict(result) for result in context.generation_results],
                "max_history_length": context.max_history_length,
                "context_window_minutes": context.context_window_minutes
            }
            
            # Write to file atomically
            temp_file = session_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(context_data, f, indent=2)
            
            # Atomic rename
            temp_file.rename(session_file)
            
            # Update cache
            with self._cache_lock:
                self._session_cache[session_id] = context
                
        except Exception as e:
            print(f"Failed to save session context for {session_id}: {e}")
    
    def _load_session_context(self, session_id: str) -> ConversationContextManager:
        """
        Load conversation context from storage or create new one.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Loaded or new ConversationContextManager
        """
        session_file = self.storage_dir / f"session_{session_id}.json"
        
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    context_data = json.load(f)
                
                # Check if session has expired
                last_updated = context_data.get("last_updated", 0)
                if time.time() - last_updated > (self.session_expiry_hours * 3600):
                    print(f"Session {session_id} expired, creating new context")
                    return ConversationContextManager()
                
                # Recreate context from saved data
                context = ConversationContextManager(
                    max_history_length=context_data.get("max_history_length", 50),
                    context_window_minutes=context_data.get("context_window_minutes", 60)
                )
                
                # Restore generation results
                for result_data in context_data.get("generation_results", []):
                    result = GenerationResult(**result_data)
                    context.generation_results.append(result)
                    
                    # Rebuild recent images cache
                    if result.result_type == "image" and "image_url" in result.result_data:
                        context._recent_images[result.request_id] = result
                
                print(f"Loaded session {session_id} with {len(context.generation_results)} results")
                return context
                
            except Exception as e:
                print(f"Failed to load session context for {session_id}: {e}")
                print("Creating new context")
        
        # Create new context if file doesn't exist or loading failed
        return ConversationContextManager()
    
    def clear_session_context(self, session_id: str) -> None:
        """
        Clear conversation context for a session.
        
        Args:
            session_id: Session identifier
        """
        # Remove from cache
        with self._cache_lock:
            if session_id in self._session_cache:
                del self._session_cache[session_id]
        
        # Remove from storage
        session_file = self.storage_dir / f"session_{session_id}.json"
        if session_file.exists():
            session_file.unlink()
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired session files.
        
        Returns:
            Number of sessions cleaned up
        """
        cleaned_count = 0
        current_time = time.time()
        expiry_threshold = self.session_expiry_hours * 3600
        
        for session_file in self.storage_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r') as f:
                    context_data = json.load(f)
                
                last_updated = context_data.get("last_updated", 0)
                if current_time - last_updated > expiry_threshold:
                    session_file.unlink()
                    cleaned_count += 1
                    
                    # Also remove from cache
                    session_id = session_file.stem.replace("session_", "")
                    with self._cache_lock:
                        if session_id in self._session_cache:
                            del self._session_cache[session_id]
                            
            except Exception as e:
                print(f"Error checking session file {session_file}: {e}")
        
        return cleaned_count
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary of session context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary information
        """
        context = self.get_session_context(session_id)
        
        return {
            "session_id": session_id,
            "total_results": len(context.generation_results),
            "recent_images": len(context.get_recent_images()),
            "has_context": len(context.generation_results) > 0,
            "last_activity": max([r.timestamp for r in context.generation_results]) if context.generation_results else None
        }
    
    def list_active_sessions(self) -> List[str]:
        """
        List all active session IDs.
        
        Returns:
            List of session IDs with non-expired contexts
        """
        sessions = []
        current_time = time.time()
        expiry_threshold = self.session_expiry_hours * 3600
        
        for session_file in self.storage_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r') as f:
                    context_data = json.load(f)
                
                last_updated = context_data.get("last_updated", 0)
                if current_time - last_updated <= expiry_threshold:
                    session_id = session_file.stem.replace("session_", "")
                    sessions.append(session_id)
                    
            except Exception:
                continue
        
        return sessions

# Global session context manager
session_context_manager = SessionContextManager()

def get_context_for_session(session_id: str) -> ConversationContextManager:
    """
    Convenience function to get conversation context for a session.
    
    Args:
        session_id: Session identifier (user ID, session token, etc.)
        
    Returns:
        ConversationContextManager for the session
    """
    return session_context_manager.get_session_context(session_id)

def save_context_for_session(session_id: str, context: ConversationContextManager) -> None:
    """
    Convenience function to save conversation context for a session.
    
    Args:
        session_id: Session identifier
        context: Conversation context to save
    """
    session_context_manager.save_session_context(session_id, context) 