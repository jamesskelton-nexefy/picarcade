# Agent Decision Logging System

## Overview

The PicArcade agentic package now includes comprehensive decision logging that tracks every decision made by each agent, providing complete transparency into the reasoning and execution process.

## What's Logged

Every agent now logs:

- **Decision Steps**: Each individual decision with reasoning
- **Input/Output Data**: What data was considered and produced
- **Confidence Scores**: Agent confidence in each decision (0.0-1.0)
- **Execution Timing**: How long each step took
- **Error Handling**: What went wrong and how it was handled
- **Metadata**: Additional context and debugging information

## Supported Agents

✅ **WorkflowOrchestrator**: Logs workflow coordination decisions  
✅ **PromptParsingAgent**: Logs parsing strategy and validation decisions  
✅ **ReferenceRetrievalAgent**: Logs search strategy and ranking decisions  
✅ **ToolFirstAgent**: Logs tool selection and workflow planning decisions  

## Decision Types

The system tracks these types of decisions:

- `TOOL_SELECTION`: Which tools to use and why
- `WORKFLOW_PLANNING`: How to structure execution
- `PROMPT_PARSING`: How to interpret user input
- `REFERENCE_RETRIEVAL`: How to find and rank images
- `RANKING`: How to prioritize results
- `FILTERING`: What to include/exclude
- `VALIDATION`: Data quality checks
- `ERROR_HANDLING`: Recovery strategies

## Quick Start

### 1. Run the Demonstration

```bash
cd packages/agentic
python decision_logging_demo.py
```

This will:
- Test all agents with sample prompts
- Show decision statistics
- Export decision data
- Display log file locations

### 2. Access Decision Data in Code

```python
from src.pic_arcade_agentic.utils.decision_logger import decision_logger

# Get all decision history
all_decisions = decision_logger.get_decision_history()

# Filter by specific agent
prompt_decisions = decision_logger.get_decision_history("PromptParsingAgent")

# Get statistics
stats = decision_logger.get_decision_stats()
print(f"Total decisions: {stats['total_decisions']}")
print(f"Success rate: {stats['successful_decisions'] / stats['total_decisions']:.2%}")

# Export to JSON
export_path = decision_logger.export_decisions_to_json()
```

## Log File Locations

### Structured Logs (JSONL)
```
packages/agentic/logs/decisions/decisions_YYYYMMDD_HHMMSS.jsonl
```
Each line is a JSON object representing one decision step.

### Exported Data (JSON)
```
packages/agentic/logs/decisions/decision_export_YYYYMMDD_HHMMSS.json
```
Complete decision data with statistics and metadata.

## Example Decision Log Entry

```json
{
  "step_id": "parse_1703123456_0",
  "timestamp": 1703123456.789,
  "agent_name": "PromptParsingAgent",
  "decision_type": "prompt_parsing",
  "input_data": {
    "prompt": "Create a portrait of Emma Stone in Renaissance style",
    "strategy": "gpt4o_structured_extraction"
  },
  "decision_reasoning": "Using GPT-4o with structured JSON response to extract intent, entities, modifiers, and references from 45 character prompt",
  "output_data": {
    "intent": "generate_portrait",
    "entities_extracted": 2,
    "modifiers_extracted": 1,
    "references_extracted": 2,
    "overall_confidence": 0.92
  },
  "confidence_score": 0.92,
  "execution_time_ms": 1247.3,
  "metadata": {
    "parsing_success": true,
    "ready_for_downstream": true
  }
}
```

## Using Decision Data for Analysis

### 1. Performance Analysis
```python
# Analyze execution times by agent
history = decision_logger.get_decision_history()
agent_performance = {}

for decision in history:
    agent = decision.agent_name
    if decision.completed_at:
        duration = (decision.completed_at - decision.started_at) * 1000
        if agent not in agent_performance:
            agent_performance[agent] = []
        agent_performance[agent].append(duration)

for agent, times in agent_performance.items():
    avg_time = sum(times) / len(times)
    print(f"{agent}: {avg_time:.2f}ms average")
```

### 2. Error Analysis
```python
# Find all failed decisions
failed_decisions = [d for d in history if not d.success]

# Group by error type
error_types = {}
for decision in failed_decisions:
    for step in decision.steps:
        if step.error:
            error_type = step.metadata.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1

print("Error frequency:", error_types)
```

### 3. Confidence Analysis
```python
# Analyze confidence scores
confidences = []
for decision in history:
    for step in decision.steps:
        if step.confidence_score is not None:
            confidences.append(step.confidence_score)

if confidences:
    avg_confidence = sum(confidences) / len(confidences)
    print(f"Average confidence: {avg_confidence:.2f}")
```

## Integration with Your Code

### Adding Decision Logging to New Agents

```python
from ..utils.decision_logger import decision_logger, DecisionType

class MyCustomAgent:
    async def process_request(self, request):
        # Start decision tracking
        request_id = f"custom_{int(time.time() * 1000)}"
        decision_logger.start_decision(
            request_id=request_id,
            agent_name="MyCustomAgent",
            initial_context={"request": request}
        )
        
        try:
            # Log a decision step
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.TOOL_SELECTION,
                input_data={"request": request},
                decision_reasoning="Analyzing request to select appropriate processing tool",
                output_data={"selected_tool": "my_tool"},
                confidence_score=0.85,
                metadata={"processing_method": "custom"}
            )
            
            # ... your processing logic ...
            
            # Complete decision tracking
            decision_logger.complete_decision(
                request_id=request_id,
                final_result={"success": True},
                success=True
            )
            
        except Exception as e:
            # Log errors
            decision_logger.complete_decision(
                request_id=request_id,
                final_result={"error": str(e)},
                success=False
            )
            raise
```

## Configuration

### Log Level
```python
import logging
from src.pic_arcade_agentic.utils.decision_logger import DecisionLogger

# Create logger with custom settings
custom_logger = DecisionLogger(
    log_level=logging.DEBUG,  # More detailed logging
    enable_file_logging=True,
    log_directory="/custom/path/logs"
)
```

### Disable Logging
```python
# Disable file logging for performance
logger = DecisionLogger(enable_file_logging=False)
```

## Best Practices

### 1. Meaningful Decision Reasoning
```python
# Good: Specific reasoning
decision_reasoning="Using GPT-4o with temperature 0.1 for consistent parsing of 45-character prompt"

# Avoid: Vague reasoning  
decision_reasoning="Processing prompt"
```

### 2. Appropriate Confidence Scores
- `0.9-1.0`: Very confident (deterministic operations)
- `0.7-0.9`: Confident (good data, clear logic)
- `0.5-0.7`: Moderate (some uncertainty)
- `0.3-0.5`: Low confidence (high uncertainty)
- `0.0-0.3`: Very uncertain or error conditions

### 3. Useful Metadata
```python
metadata={
    "model_used": "gpt-4o",
    "api_provider": "openai", 
    "retry_count": 0,
    "processing_strategy": "structured_extraction"
}
```

## Troubleshooting

### Missing Log Files
- Check that the `packages/agentic/logs/decisions/` directory exists
- Verify write permissions
- Check for disk space

### Performance Impact
- Decision logging adds ~1-5ms per decision step
- File logging can be disabled for production if needed
- Log files are rotated automatically by timestamp

### Memory Usage
- Completed decisions are stored in memory
- Use `decision_logger.completed_decisions.clear()` to free memory
- Export and clear periodically for long-running processes

## Future Enhancements

- [ ] Log retention policies
- [ ] Real-time dashboard
- [ ] Decision replay functionality
- [ ] ML-based decision quality scoring
- [ ] Integration with monitoring systems

## Support

For questions about decision logging:
1. Check the demonstration script: `decision_logging_demo.py`
2. Review log files in `logs/decisions/`
3. Use `decision_logger.get_decision_stats()` for debugging 