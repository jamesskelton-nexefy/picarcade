#!/usr/bin/env python3
"""
Simple Tool-First Architecture Demonstration

Shows the tool-first architecture concepts and tool registry functionality
without requiring API keys.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pic_arcade_agentic.tools.base import Tool, ToolRegistry, ToolResult, ToolCategory


class MockPromptParsingTool(Tool):
    """Mock prompt parsing tool for demonstration."""
    
    def __init__(self):
        super().__init__(
            name="mock_prompt_parser",
            description="Parse user prompts (mock implementation)",
            category=ToolCategory.PROMPT_PROCESSING,
            input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"intent": {"type": "string"}}}
        )
    
    def _validate_config(self) -> None:
        pass
    
    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        prompt = input_data.get("prompt", "")
        
        # Mock parsing logic
        if "portrait" in prompt.lower():
            intent = "generate_portrait"
        elif "landscape" in prompt.lower():
            intent = "generate_landscape"
        elif "face swap" in prompt.lower():
            intent = "face_swap"
        else:
            intent = "generate_image"
        
        return ToolResult(
            success=True,
            data={
                "intent": intent,
                "entities": [{"text": "mock entity", "type": "person", "confidence": 0.9}],
                "confidence": 0.85
            }
        )


class MockSearchTool(Tool):
    """Mock search tool for demonstration."""
    
    def __init__(self):
        super().__init__(
            name="mock_search",
            description="Search for reference images (mock implementation)",
            category=ToolCategory.IMAGE_SEARCH,
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"images": {"type": "array"}}}
        )
    
    def _validate_config(self) -> None:
        pass
    
    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        query = input_data.get("query", "")
        
        # Mock search results
        mock_images = [
            {
                "url": f"https://example.com/image1_{query.replace(' ', '_')}.jpg",
                "title": f"Reference image for {query}",
                "rank_score": 0.95
            },
            {
                "url": f"https://example.com/image2_{query.replace(' ', '_')}.jpg",
                "title": f"Alternative reference for {query}",
                "rank_score": 0.87
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "images": mock_images,
                "total_results": len(mock_images),
                "summary": f"Found {len(mock_images)} reference images for '{query}'"
            }
        )


class MockImageGenerationTool(Tool):
    """Mock image generation tool for demonstration."""
    
    def __init__(self):
        super().__init__(
            name="mock_image_generation",
            description="Generate images using AI (mock implementation)",
            category=ToolCategory.IMAGE_GENERATION,
            input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"images": {"type": "array"}}}
        )
    
    def _validate_config(self) -> None:
        pass
    
    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        prompt = input_data.get("prompt", "")
        
        # Mock generation results
        mock_images = [
            {
                "url": f"https://example.com/generated_{prompt.replace(' ', '_')}_001.jpg",
                "width": 1024,
                "height": 1024,
                "seed": 12345
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "images": mock_images,
                "generation_time": 15.2,
                "model_version": "mock-flux-pro"
            }
        )


class MockWorkflowPlanner(Tool):
    """Mock workflow planner for demonstration."""
    
    def __init__(self):
        super().__init__(
            name="mock_workflow_planner",
            description="Plan multi-step workflows (mock implementation)",
            category=ToolCategory.WORKFLOW_PLANNING,
            input_schema={"type": "object", "properties": {"user_request": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"workflow_plan": {"type": "array"}}}
        )
    
    def _validate_config(self) -> None:
        pass
    
    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        user_request = input_data.get("user_request", "")
        
        # Mock workflow planning
        if "portrait" in user_request.lower():
            workflow = [
                {"step": 1, "tool_name": "mock_prompt_parser", "description": "Parse the prompt"},
                {"step": 2, "tool_name": "mock_search", "description": "Find reference images"},
                {"step": 3, "tool_name": "mock_image_generation", "description": "Generate portrait"}
            ]
        elif "face swap" in user_request.lower():
            workflow = [
                {"step": 1, "tool_name": "mock_prompt_parser", "description": "Parse the prompt"},
                {"step": 2, "tool_name": "mock_search", "description": "Find source images"},
                {"step": 3, "tool_name": "mock_face_swap", "description": "Perform face swap"},
                {"step": 4, "tool_name": "mock_quality_check", "description": "Assess result quality"}
            ]
        else:
            workflow = [
                {"step": 1, "tool_name": "mock_prompt_parser", "description": "Parse the prompt"},
                {"step": 2, "tool_name": "mock_image_generation", "description": "Generate image"}
            ]
        
        return ToolResult(
            success=True,
            data={
                "workflow_plan": workflow,
                "reasoning": f"Planned {len(workflow)} steps for request: {user_request}",
                "confidence": 0.92,
                "estimated_time": len(workflow) * 10
            }
        )


async def demonstrate_tool_registry():
    """Demonstrate the tool registry functionality."""
    print("🔧 TOOL REGISTRY DEMONSTRATION")
    print("=" * 50)
    
    # Create registry and tools
    registry = ToolRegistry()
    
    tools = [
        MockPromptParsingTool(),
        MockSearchTool(),
        MockImageGenerationTool(),
        MockWorkflowPlanner()
    ]
    
    # Register tools
    for tool in tools:
        registry.register(tool)
    
    print(f"✅ Registered {len(tools)} tools")
    
    # Show available tools
    print("\n🛠️  Available Tools:")
    for tool_name in registry.list_all_tools():
        tool = registry.get_tool(tool_name)
        print(f"  - {tool.name}: {tool.description}")
    
    # Demonstrate tool discovery
    print("\n🔍 Tool Discovery:")
    search_tools = registry.search_tools("search")
    print(f"  Found {len(search_tools)} tools matching 'search'")
    
    generation_tools = registry.get_tools_by_category(ToolCategory.IMAGE_GENERATION)
    print(f"  Found {len(generation_tools)} image generation tools")
    
    return registry


async def demonstrate_dynamic_workflow(registry: ToolRegistry):
    """Demonstrate dynamic workflow planning and execution."""
    print("\n\n🚀 DYNAMIC WORKFLOW DEMONSTRATION")
    print("=" * 50)
    
    test_requests = [
        "Create a portrait of Emma Stone",
        "Generate a landscape painting",
        "Perform face swap between two images",
        "Make an anime-style character"
    ]
    
    planner = registry.get_tool("mock_workflow_planner")
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n{i}. Request: '{request}'")
        print("-" * 40)
        
        # Plan workflow
        result = await planner.invoke({"user_request": request})
        
        if result.success:
            workflow = result.data["workflow_plan"]
            print(f"   📋 Planned Workflow ({len(workflow)} steps):")
            
            for step in workflow:
                print(f"      {step['step']}. {step['tool_name']}: {step['description']}")
            
            print(f"   ⏱️  Estimated Time: {result.data.get('estimated_time', 'N/A')}s")
            print(f"   🎯 Confidence: {result.data.get('confidence', 'N/A')}")
        else:
            print(f"   ❌ Planning failed: {result.error}")


async def demonstrate_tool_execution(registry: ToolRegistry):
    """Demonstrate individual tool execution."""
    print("\n\n⚡ TOOL EXECUTION DEMONSTRATION")
    print("=" * 50)
    
    # Test prompt parsing
    print("\n1. Prompt Parsing:")
    parser = registry.get_tool("mock_prompt_parser")
    result = await parser.invoke({"prompt": "Create a portrait of Taylor Swift"})
    
    if result.success:
        print(f"   Intent: {result.data['intent']}")
        print(f"   Confidence: {result.data['confidence']}")
    
    # Test search
    print("\n2. Image Search:")
    search = registry.get_tool("mock_search")
    result = await search.invoke({"query": "Taylor Swift portrait"})
    
    if result.success:
        print(f"   Found {result.data['total_results']} images")
        print(f"   Summary: {result.data['summary']}")
    
    # Test generation
    print("\n3. Image Generation:")
    generator = registry.get_tool("mock_image_generation")
    result = await generator.invoke({"prompt": "Portrait of Taylor Swift"})
    
    if result.success:
        print(f"   Generated {len(result.data['images'])} image(s)")
        print(f"   Generation time: {result.data['generation_time']}s")
        print(f"   Model: {result.data['model_version']}")


async def show_architecture_benefits():
    """Show the benefits of tool-first architecture."""
    print("\n\n✅ TOOL-FIRST ARCHITECTURE BENEFITS")
    print("=" * 50)
    
    benefits = [
        ("🔄 Dynamic Planning", "Workflows adapt to different request types"),
        ("🧩 Modularity", "Tools can be mixed and matched"),
        ("🔌 Extensibility", "New tools are automatically discovered"),
        ("🎯 Intelligence", "AI selects optimal tool sequences"),
        ("🔧 Maintainability", "Tools are isolated and testable"),
        ("⚡ Performance", "Tools can be cached and optimized"),
        ("🔍 Discoverability", "Tools self-describe their capabilities"),
        ("🔀 Flexibility", "Same tools work in different workflows")
    ]
    
    for emoji_title, description in benefits:
        print(f"  {emoji_title}: {description}")
    
    print("\n💡 Key Insight:")
    print("   Instead of hardcoded workflows, agents reason about")
    print("   which tools to use and how to chain them together!")


async def compare_approaches():
    """Compare old vs new approach."""
    print("\n\n📊 OLD vs NEW APPROACH COMPARISON")
    print("=" * 50)
    
    comparison = [
        ("Request Handling", "Fixed patterns only", "Any request type"),
        ("Tool Selection", "Hardcoded", "AI-powered"),
        ("Extensibility", "Code changes", "Drop-in tools"),
        ("Workflow", "Static pipeline", "Dynamic planning"),
        ("Maintainability", "Tightly coupled", "Modular design"),
        ("Testing", "Integration only", "Unit + Integration"),
        ("Error Handling", "Pipeline breaks", "Graceful degradation"),
        ("Discovery", "Manual", "Automatic")
    ]
    
    print(f"{'Aspect':<20} {'Old Approach':<20} {'New Approach':<20}")
    print("-" * 62)
    
    for aspect, old, new in comparison:
        print(f"{aspect:<20} {old:<20} {new:<20}")


async def main():
    """Run the complete demonstration."""
    print("🚀 PIC ARCADE: TOOL-FIRST ARCHITECTURE (SIMPLE DEMO)")
    print("Demonstrating modular, intelligent tool selection")
    print("=" * 70)
    
    try:
        # Demonstrate core concepts
        registry = await demonstrate_tool_registry()
        await demonstrate_dynamic_workflow(registry)
        await demonstrate_tool_execution(registry)
        await show_architecture_benefits()
        await compare_approaches()
        
        print("\n\n🎉 DEMONSTRATION COMPLETE!")
        print("\nThis shows how the tool-first architecture transforms")
        print("Pic Arcade into a modular, intelligent AI platform that")
        print("can dynamically adapt to any user request!")
        
        print("\n🔗 Next Steps:")
        print("  1. Add your API keys to .env file")
        print("  2. Run: npm run demo:tool-first")
        print("  3. See the full system in action!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 