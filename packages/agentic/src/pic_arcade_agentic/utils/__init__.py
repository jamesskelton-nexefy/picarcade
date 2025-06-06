"""
Utility modules for PicArcade Agentic package.
"""

from .decision_logger import DecisionLogger, DecisionStep, AgentDecision
from .conversation_context import ConversationContextManager, GenerationResult, conversation_context

# Mem0 integration (optional import)
try:
    from .mem0_context import (
        Mem0ConversationContext, 
        get_mem0_context, 
        create_user_session_id
    )
    HAS_MEM0 = True
except ImportError:
    HAS_MEM0 = False

# Session context (file-based fallback)
try:
    from .session_context import (
        SessionContextManager, 
        session_context_manager,
        get_context_for_session,
        save_context_for_session
    )
    HAS_SESSION_CONTEXT = True
except ImportError:
    HAS_SESSION_CONTEXT = False

__all__ = [
    'DecisionLogger', 'DecisionStep', 'AgentDecision',
    'ConversationContextManager', 'GenerationResult', 'conversation_context'
]

# Add Mem0 exports if available
if HAS_MEM0:
    __all__.extend([
        'Mem0ConversationContext', 
        'get_mem0_context', 
        'create_user_session_id'
    ])

# Add session context exports if available
if HAS_SESSION_CONTEXT:
    __all__.extend([
        'SessionContextManager', 
        'session_context_manager',
        'get_context_for_session',
        'save_context_for_session'
    ]) 