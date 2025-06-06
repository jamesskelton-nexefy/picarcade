# Tool-First Architecture for Pic Arcade

This document explains how Pic Arcade can adopt the tool-first architecture pattern used by leading AI systems like Claude, Cursor, and Perplexity.

## 🎯 Overview

The tool-first architecture transforms our current hardcoded agent pipeline into a dynamic, modular system where tools are first-class citizens. Instead of fixed workflows, agents intelligently select and chain tools based on user requests.

## 🔄 Transformation Summary

### **Before: Hardcoded Agents (Phase 2)**
```python
# Fixed pipeline
PromptParsingAgent → ReferenceRetrievalAgent → WorkflowOrchestrator
```

### **After: Tool-First Architecture**
```python
# Dynamic tool selection
ToolFirstAgent → WorkflowPlanner → ToolSelector → ToolExecutor
```

## 🏗️ Architecture Components

### 1. Tool Abstraction (`tools/base.py`)

Every API integration becomes a standardized tool:

```python
class Tool(ABC):
    name: str
    description: str  
    category: ToolCategory
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    
    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult
```

### 2. Tool Registry (`tools/base.py`)

Central registry for tool discovery and invocation:

```python
tool_registry = ToolRegistry()
tool_registry.register(PromptParsingTool())
tool_registry.register(BingImageSearchTool())
# Agents can now discover and use any registered tool
```

### 3. Tool Categories

Organized by functionality:
- `PROMPT_PROCESSING`: GPT-4o parsing, optimization
- `IMAGE_SEARCH`: Bing, Google image search
- `IMAGE_GENERATION`: Flux, DALL-E generation
- `IMAGE_EDITING`: Inpainting, outpainting
- `FACE_MANIPULATION`: Face swap, enhancement
- `QUALITY_ASSESSMENT`: CLIP scoring, artifact detection
- `WORKFLOW_PLANNING`: Dynamic workflow creation

### 4. Dynamic Workflow Planning

GPT-4o plans workflows based on available tools:

```python
# User request: "Put me in Taylor Swift's dress"
# Planned workflow:
[
  {"step": 1, "tool_name": "web_search", "inputs": {...}},
  {"step": 2, "tool_name": "bing_image_search", "inputs": {...}},
  {"step": 3, "tool_name": "flux_image_generation", "inputs": {...}},
  {"step": 4, "tool_name": "face_swap", "inputs": {...}}
]
```

## 🚀 Implementation Guide

### Phase 1: Create Tool System (✅ Complete)

1. **Base Tool Classes** - Abstract tool interface
2. **Tool Registry** - Central tool management
3. **Tool Result Types** - Standardized outputs

### Phase 2: Convert Existing Agents to Tools (✅ Complete)

1. **PromptParsingTool** - Wraps GPT-4o parsing
2. **BingImageSearchTool** - Wraps Bing Search API
3. **WorkflowPlanningTool** - Dynamic workflow creation
4. **WorkflowExecutorTool** - Tool chain execution

### Phase 3: Add Generation Tools (Next)

1. **FluxImageGenerationTool** - Replicate Flux API
2. **DALLEImageGenerationTool** - OpenAI DALL-E API
3. **ImageEditingTool** - Inpainting/outpainting
4. **QualityAssessmentTool** - CLIP scoring

### Phase 4: Advanced Tools

1. **FaceSwapTool** - Face manipulation
2. **VideoGenerationTool** - Runway integration
3. **StyleTransferTool** - Art style application
4. **AdaptiveWorkflowTool** - Self-modifying workflows

## 🔧 Usage Examples

### Basic Tool Invocation
```python
# Direct tool usage
tool_registry = ToolRegistry()
parser = tool_registry.get_tool("prompt_parser")
result = await parser.invoke({"prompt": "Emma Stone portrait"})
```

### Dynamic Workflow
```python
# Agent selects tools automatically
agent = ToolFirstAgent()
result = await agent.process_request("Create Van Gogh style portrait")
# Agent plans: prompt_parser → image_search → flux_generation
```

### Tool Discovery
```python
# Find tools by capability
search_tools = tool_registry.search_tools("image search")
generation_tools = tool_registry.get_tools_by_category(ToolCategory.IMAGE_GENERATION)
```

## 📊 Benefits Over Current Approach

| Aspect | Old Approach | Tool-First |
|--------|-------------|------------|
| **Flexibility** | Fixed pipeline | Dynamic planning |
| **Extensibility** | Code changes required | Drop-in tools |
| **Request Types** | Limited patterns | Any request |
| **Tool Discovery** | Hardcoded | Automatic |
| **Error Handling** | Pipeline breaks | Graceful degradation |
| **Testing** | Integration only | Unit + Integration |
| **Maintainability** | Coupled code | Modular tools |

## 🎯 Real-World Examples

### Example 1: Celebrity Portrait
**Request:** "Create a portrait of Emma Stone in Renaissance style"

**Old Approach:** Fixed parse → search → finalize
**Tool-First:** Dynamic planning selects optimal tools and parameters

### Example 2: Complex Face Swap
**Request:** "Put me in Taylor Swift's Grammy dress"

**Old Approach:** Can't handle this request
**Tool-First:** Plans: web_search → image_search → generate_base → face_swap

### Example 3: Quality Optimization
**Request:** "Improve the quality of this image"

**Old Approach:** Not supported
**Tool-First:** Plans: quality_assessment → image_enhancement → artifact_removal

## 🔄 Migration Strategy

### Step 1: Parallel Implementation
- Keep existing Phase 2 agents working
- Build tool-first system alongside
- Compare results and performance

### Step 2: Gradual Transition
- Start with simple requests in tool-first
- Migrate complex workflows incrementally
- Maintain backward compatibility

### Step 3: Full Migration
- Replace old agents with tool-first
- Remove legacy code
- Optimize tool performance

## 🧪 Testing Strategy

### Unit Testing
```python
# Test individual tools
async def test_prompt_parsing_tool():
    tool = PromptParsingTool()
    result = await tool.invoke({"prompt": "test"})
    assert result.success
```

### Integration Testing
```python
# Test tool chains
async def test_workflow_execution():
    plan = [{"step": 1, "tool_name": "prompt_parser", ...}]
    executor = WorkflowExecutorTool()
    result = await executor.invoke({"workflow_plan": plan})
    assert result.success
```

### End-to-End Testing
```python
# Test complete agent workflows
async def test_agent_request():
    agent = ToolFirstAgent()
    result = await agent.process_request("complex request")
    assert result["success"]
```

## 🚀 Running the Demo

See the tool-first architecture in action:

```bash
# Install dependencies
npm run install:agentic

# Set API keys
export OPENAI_API_KEY="your_key"
export BING_API_KEY="your_key"

# Run the demonstration
npm run demo:tool-first
```

The demo shows:
1. **Old vs New Approach** - Side-by-side comparison
2. **Dynamic Tool Selection** - How agents choose tools
3. **Complex Workflow Planning** - Multi-step tool chains
4. **Extensibility** - How easy it is to add new tools

## 🎯 Next Steps

1. **Test the Tool-First Demo** - Run and explore the new architecture
2. **Phase 3 Planning** - Design Flux and image generation tools
3. **Performance Benchmarking** - Compare with old approach
4. **Production Migration** - Plan the transition strategy

## 🤝 Contributing

When adding new tools:

1. **Inherit from Tool** - Use the base class
2. **Define Schemas** - Clear input/output specifications
3. **Add Tests** - Unit and integration tests
4. **Register Tool** - Add to the global registry
5. **Update Documentation** - Explain the tool's purpose

The tool-first architecture makes Pic Arcade more intelligent, modular, and extensible - exactly what's needed for a production AI platform.

---

**🔗 Related Files:**
- `packages/agentic/src/pic_arcade_agentic/tools/` - Tool implementations
- `packages/agentic/src/pic_arcade_agentic/agents/tool_agent.py` - Tool-first agent
- `packages/agentic/examples/tool_first_demo.py` - Live demonstration
- `.cursor/rules/docs/tooluse.md` - Original inspiration guide 