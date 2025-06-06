"""
Decision Logger for Agent Decision Tracking

Provides comprehensive logging and auditing of agent decisions, reasoning,
and intermediate results for review and debugging.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import os

class DecisionType(Enum):
    """Types of decisions that can be logged."""
    TOOL_SELECTION = "tool_selection"
    WORKFLOW_PLANNING = "workflow_planning"
    PROMPT_PARSING = "prompt_parsing"
    REFERENCE_RETRIEVAL = "reference_retrieval"
    RANKING = "ranking"
    FILTERING = "filtering"
    VALIDATION = "validation"
    ERROR_HANDLING = "error_handling"

@dataclass
class DecisionStep:
    """Represents a single decision step with context and reasoning."""
    step_id: str
    timestamp: float
    agent_name: str
    decision_type: DecisionType
    input_data: Dict[str, Any]
    decision_reasoning: str
    output_data: Dict[str, Any]
    confidence_score: Optional[float] = None
    execution_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class AgentDecision:
    """Container for a complete agent decision with all steps."""
    request_id: str
    agent_name: str
    started_at: float
    completed_at: Optional[float] = None
    total_steps: int = 0
    steps: List[DecisionStep] = None
    final_result: Optional[Dict[str, Any]] = None
    success: bool = True
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []

class DecisionLogger:
    """
    Centralized logger for tracking agent decisions and reasoning.
    
    Provides structured logging, JSON export, and decision analytics
    for comprehensive agent behavior auditing.
    """
    
    def __init__(
        self, 
        log_level: int = logging.INFO,
        enable_file_logging: bool = True,
        log_directory: Optional[str] = None
    ):
        """Initialize the decision logger."""
        self.logger = logging.getLogger(f"decision_logger.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Track active decisions
        self.active_decisions: Dict[str, AgentDecision] = {}
        self.completed_decisions: List[AgentDecision] = []
        
        # File logging setup
        self.enable_file_logging = enable_file_logging
        self.log_directory = log_directory or self._get_default_log_directory()
        
        if self.enable_file_logging:
            self._setup_file_logging()
        
        # Decision analytics
        self.decision_stats = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "average_execution_time": 0.0,
            "decision_types": {}
        }
    
    def _get_default_log_directory(self) -> str:
        """Get default log directory in the agentic package."""
        package_dir = Path(__file__).parent.parent.parent.parent
        log_dir = package_dir / "logs" / "decisions"
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir)
    
    def _setup_file_logging(self) -> None:
        """Setup file logging for decisions."""
        try:
            # Create timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(self.log_directory) / f"decisions_{timestamp}.jsonl"
            
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Create formatter for structured logging
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.info(f"Decision logging enabled to: {log_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup file logging: {e}")
            self.enable_file_logging = False
    
    def start_decision(
        self, 
        request_id: str, 
        agent_name: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracking a new agent decision.
        
        Args:
            request_id: Unique identifier for this request
            agent_name: Name of the agent making decisions
            initial_context: Optional initial context data
            
        Returns:
            Decision ID for referencing this decision
        """
        decision = AgentDecision(
            request_id=request_id,
            agent_name=agent_name,
            started_at=time.time()
        )
        
        self.active_decisions[request_id] = decision
        
        self.logger.info(
            f"Started decision tracking for {agent_name} (request: {request_id})",
            extra={
                "decision_event": "start",
                "request_id": request_id,
                "agent_name": agent_name,
                "initial_context": initial_context or {}
            }
        )
        
        return request_id
    
    def log_decision_step(
        self,
        request_id: str,
        decision_type: DecisionType,
        input_data: Dict[str, Any],
        decision_reasoning: str,
        output_data: Dict[str, Any],
        confidence_score: Optional[float] = None,
        execution_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log a decision step with detailed context.
        
        Args:
            request_id: Request ID from start_decision
            decision_type: Type of decision being made
            input_data: Input data for this decision
            decision_reasoning: Explanation of why this decision was made
            output_data: Result of the decision
            confidence_score: Optional confidence in the decision (0.0-1.0)
            execution_time_ms: Time taken to make this decision
            metadata: Additional metadata for this step
            error: Error message if decision failed
        """
        if request_id not in self.active_decisions:
            self.logger.warning(f"No active decision found for request_id: {request_id}")
            return
        
        decision = self.active_decisions[request_id]
        
        # Create decision step
        step = DecisionStep(
            step_id=f"{request_id}_{len(decision.steps)}",
            timestamp=time.time(),
            agent_name=decision.agent_name,
            decision_type=decision_type,
            input_data=input_data,
            decision_reasoning=decision_reasoning,
            output_data=output_data,
            confidence_score=confidence_score,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
            error=error
        )
        
        # Add step to decision
        decision.steps.append(step)
        decision.total_steps += 1
        
        # Update success status
        if error:
            decision.success = False
        
        # Log the step
        log_level = logging.ERROR if error else logging.INFO
        self.logger.log(
            log_level,
            f"Decision step: {decision_type.value} | {decision_reasoning}",
            extra={
                "decision_event": "step",
                "request_id": request_id,
                "step_id": step.step_id,
                "decision_type": decision_type.value,
                "reasoning": decision_reasoning,
                "confidence": confidence_score,
                "execution_time_ms": execution_time_ms,
                "input_summary": self._summarize_data(input_data),
                "output_summary": self._summarize_data(output_data),
                "error": error
            }
        )
        
        # Write to file if enabled
        if self.enable_file_logging:
            self._write_step_to_file(step)
    
    def complete_decision(
        self, 
        request_id: str, 
        final_result: Optional[Dict[str, Any]] = None,
        success: Optional[bool] = None
    ) -> AgentDecision:
        """
        Complete and finalize a decision.
        
        Args:
            request_id: Request ID to complete
            final_result: Final result data
            success: Override success status
            
        Returns:
            Completed AgentDecision object
        """
        if request_id not in self.active_decisions:
            self.logger.warning(f"No active decision found for request_id: {request_id}")
            return None
        
        decision = self.active_decisions.pop(request_id)
        decision.completed_at = time.time()
        decision.final_result = final_result
        
        if success is not None:
            decision.success = success
        
        # Add to completed decisions
        self.completed_decisions.append(decision)
        
        # Update statistics
        self._update_decision_stats(decision)
        
        # Log completion
        total_time = (decision.completed_at - decision.started_at) * 1000
        self.logger.info(
            f"Completed decision for {decision.agent_name} "
            f"({decision.total_steps} steps, {total_time:.2f}ms, "
            f"{'success' if decision.success else 'failed'})",
            extra={
                "decision_event": "complete",
                "request_id": request_id,
                "agent_name": decision.agent_name,
                "total_steps": decision.total_steps,
                "total_time_ms": total_time,
                "success": decision.success,
                "final_result_summary": self._summarize_data(final_result or {})
            }
        )
        
        return decision
    
    def _summarize_data(self, data: Dict[str, Any], max_length: int = 200) -> str:
        """Create a summary of data for logging."""
        try:
            if not data:
                return "empty"
            
            # Convert to JSON and truncate if needed
            json_str = json.dumps(data, default=str)
            if len(json_str) <= max_length:
                return json_str
            
            # Return truncated version with key info
            keys = list(data.keys())
            return f"{{keys: {keys}, length: {len(json_str)} chars}}"
            
        except Exception:
            return f"{{type: {type(data).__name__}, items: {len(data) if hasattr(data, '__len__') else 'unknown'}}}"
    
    def _write_step_to_file(self, step: DecisionStep) -> None:
        """Write decision step to JSONL file."""
        try:
            # Convert step to dict
            step_data = asdict(step)
            step_data['decision_type'] = step.decision_type.value
            
            # Write to latest log file
            log_files = list(Path(self.log_directory).glob("decisions_*.jsonl"))
            if log_files:
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                with open(latest_log, 'a') as f:
                    f.write(json.dumps(step_data, default=str) + '\n')
        except Exception as e:
            self.logger.warning(f"Failed to write step to file: {e}")
    
    def _update_decision_stats(self, decision: AgentDecision) -> None:
        """Update decision statistics."""
        self.decision_stats["total_decisions"] += 1
        
        if decision.success:
            self.decision_stats["successful_decisions"] += 1
        else:
            self.decision_stats["failed_decisions"] += 1
        
        # Update average execution time
        if decision.completed_at:
            exec_time = (decision.completed_at - decision.started_at) * 1000
            total_decisions = self.decision_stats["total_decisions"]
            current_avg = self.decision_stats["average_execution_time"]
            self.decision_stats["average_execution_time"] = (
                (current_avg * (total_decisions - 1) + exec_time) / total_decisions
            )
        
        # Update decision type counts
        for step in decision.steps:
            decision_type = step.decision_type.value
            self.decision_stats["decision_types"][decision_type] = (
                self.decision_stats["decision_types"].get(decision_type, 0) + 1
            )
    
    def get_decision_history(self, agent_name: Optional[str] = None) -> List[AgentDecision]:
        """Get completed decision history, optionally filtered by agent."""
        if agent_name:
            return [d for d in self.completed_decisions if d.agent_name == agent_name]
        return self.completed_decisions.copy()
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision statistics."""
        return self.decision_stats.copy()
    
    def export_decisions_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export all decision data to JSON file.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to exported file
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(self.log_directory) / f"decision_export_{timestamp}.json"
        
        export_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_decisions": len(self.completed_decisions),
                "active_decisions": len(self.active_decisions)
            },
            "statistics": self.decision_stats,
            "completed_decisions": [
                self._decision_to_dict(decision) 
                for decision in self.completed_decisions
            ],
            "active_decisions": [
                self._decision_to_dict(decision) 
                for decision in self.active_decisions.values()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported decision data to: {filepath}")
        return str(filepath)
    
    def _decision_to_dict(self, decision: AgentDecision) -> Dict[str, Any]:
        """Convert AgentDecision to dictionary for export."""
        decision_dict = asdict(decision)
        
        # Convert decision types to strings
        for step in decision_dict.get('steps', []):
            if 'decision_type' in step:
                step['decision_type'] = step['decision_type'].value if hasattr(step['decision_type'], 'value') else str(step['decision_type'])
        
        return decision_dict

# Global decision logger instance
decision_logger = DecisionLogger() 