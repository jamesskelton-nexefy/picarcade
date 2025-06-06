#!/usr/bin/env python3
"""
Tool-First Architecture Demonstration

Shows the transformation from hardcoded agents to dynamic tool selection
with advanced Flux Kontext Max capabilities for professional image editing.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from pic_arcade_agentic.agents.tool_agent import ToolFirstAgent
from pic_arcade_agentic.agents.prompt_parser import PromptParsingAgent
from pic_arcade_agentic.agents.reference_retriever import ReferenceRetrievalAgent
from pic_arcade_agentic.workflow.orchestrator import WorkflowOrchestrator


async def demo_old_approach():
    """Demonstrate the old hardcoded approach (Phase 2)."""
    print("📊 OLD APPROACH: Hardcoded Agents (Phase 2)")
    print("=" * 60)
    
    # Hardcoded workflow with specific agents
    prompt = "Create a portrait of Emma Stone in Van Gogh style"
    
    try:
        # Step 1: Parse prompt (hardcoded agent)
        parser = PromptParsingAgent()
        parsed_result = await parser.parse_prompt(prompt)
        
        print(f"✅ Parsed prompt with {len(parsed_result.entities)} entities")
        print(f"   Intent: {parsed_result.intent}")
        print(f"   References: {[r.text for r in parsed_result.references]}")
        
        # Step 2: Retrieve references (hardcoded agent) - Skip if no Bing API
        if parsed_result.references and os.getenv("BING_API_KEY"):
            retriever = ReferenceRetrievalAgent()
            updated_refs = await retriever.retrieve_references(parsed_result.references)
            
            total_images = sum(len(ref.image_urls) for ref in updated_refs)
            print(f"✅ Retrieved {total_images} reference images")
        elif parsed_result.references:
            print("⚠️  Skipped reference retrieval (no Bing API)")
        
        # Step 3: Fixed orchestration (LangGraph)
        if os.getenv("BING_API_KEY"):
            orchestrator = WorkflowOrchestrator()
            final_state = await orchestrator.process_prompt(prompt)
            print(f"✅ Workflow completed: {final_state.status}")
        else:
            print("⚠️  Skipped workflow orchestration (requires Bing API)")
        
        print("\n❌ LIMITATIONS OF OLD APPROACH:")
        print("   - Fixed workflow steps (parse → retrieve → finalize)")
        print("   - Can't adapt to different request types")
        print("   - Adding new capabilities requires code changes")
        print("   - No dynamic tool selection")
        print("   - Requires specific APIs (Bing for search)")
        print("   - No advanced editing capabilities")
        
    except Exception as e:
        print(f"❌ Old approach failed: {e}")
        print("   This demonstrates the brittleness of hardcoded workflows")


async def demo_tool_first_approach():
    """Demonstrate the new tool-first approach with Flux Kontext Max."""
    print("\n\n🔧 NEW APPROACH: Tool-First Architecture with Flux Kontext Max")
    print("=" * 70)
    
    agent = ToolFirstAgent()
    
    # Show available tools with focus on Flux capabilities
    tools = agent.get_available_tools()
    print(f"🛠️  Available Tools: {len(tools)}")
    
    # Group tools by category for better presentation
    tool_categories = {}
    for tool in tools:
        category = tool.get('category', 'Other')
        if category not in tool_categories:
            tool_categories[category] = []
        tool_categories[category].append(tool)
    
    for category, category_tools in tool_categories.items():
        print(f"\n📂 {category.replace('_', ' ').title()}:")
        for tool in category_tools:
            print(f"   - {tool['name']}: {tool['description'][:80]}...")
    
    # Test Flux Kontext Max specific requests
    flux_requests = [
        "Transform this portrait to watercolor painting style",
        "Change the hair color to blonde in this photo", 
        "Replace the text on this sign with 'PIC ARCADE'",
        "Change the background to a beach scene",
        "Generate the same character in different poses"
    ]
    
    print(f"\n🎨 Processing {len(flux_requests)} Flux Kontext Max requests:")
    
    for i, request in enumerate(flux_requests, 1):
        print(f"\n{i}. Request: {request}")
        print("-" * 50)
        
        result = await agent.process_request(request)
        
        if result["success"]:
            workflow = result["workflow_plan"]
            print(f"   📋 Planned Steps: {len(workflow['workflow_plan'])}")
            
            for step in workflow['workflow_plan']:
                tool_name = step['tool_name']
                # Highlight Flux tools
                prefix = "🎨" if "flux" in tool_name.lower() or any(x in tool_name.lower() for x in ["style", "object", "text", "background", "character"]) else "🔧"
                print(f"      {step['step']}. {prefix} {tool_name}: {step['description']}")
            
            print(f"   ⏱️  Estimated Time: {workflow.get('estimated_time', 'N/A')}s")
            print(f"   🎯 Confidence: {workflow.get('confidence', 'N/A')}")
            print(f"   🔗 Tools Used: {result['metadata']['tools_used']}")
            
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n✅ BENEFITS OF TOOL-FIRST APPROACH:")
    print("   ✓ Dynamic workflow planning based on request")
    print("   ✓ Modular, reusable tools")
    print("   ✓ Professional image editing capabilities")
    print("   ✓ Easy to add new capabilities")
    print("   ✓ Intelligent tool selection and chaining")
    print("   ✓ Adapts to diverse request types")


async def demonstrate_flux_capabilities():
    """Showcase specific Flux Kontext Max capabilities."""
    print("\n\n🎨 FLUX KONTEXT MAX CAPABILITIES DEMONSTRATION")
    print("=" * 60)
    
    agent = ToolFirstAgent()
    
    # Professional editing scenarios
    scenarios = [
        {
            "name": "Professional Portrait Enhancement",
            "request": "Transform this business headshot to digital art style, change the suit to navy blue, and add a modern office background",
            "expected_tools": ["style_transfer", "object_change", "background_swap"]
        },
        {
            "name": "Brand Marketing Campaign",
            "request": "Replace the logo on this billboard with 'PIC ARCADE', convert to vintage style, and create watercolor version",
            "expected_tools": ["text_editing", "style_transfer"]
        },
        {
            "name": "Character Design Consistency",
            "request": "Generate this character sitting, standing, and running while maintaining the same facial features and clothing",
            "expected_tools": ["character_consistency"]
        },
        {
            "name": "Creative Content Transformation",
            "request": "Change the hair to blonde curly, replace background with magical forest, and convert to oil painting style",
            "expected_tools": ["object_change", "background_swap", "style_transfer"]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. 🎭 {scenario['name']}")
        print(f"   Request: {scenario['request']}")
        print("-" * 50)
        
        result = await agent.process_request(scenario['request'])
        
        if result["success"]:
            workflow = result["workflow_plan"]
            planned_tools = [step['tool_name'] for step in workflow['workflow_plan']]
            
            # Check if expected tools are included
            expected = scenario['expected_tools']
            found_expected = [tool for tool in expected if any(tool in planned_tool for planned_tool in planned_tools)]
            
            print(f"   📋 Planned workflow ({len(workflow['workflow_plan'])} steps):")
            for step in workflow['workflow_plan']:
                tool_name = step['tool_name']
                is_flux = any(x in tool_name.lower() for x in ["flux", "style", "object", "text", "background", "character"])
                prefix = "🎨" if is_flux else "🔧"
                print(f"      {step['step']}. {prefix} {tool_name}")
                print(f"         └─ {step['description']}")
            
            print(f"   🎯 Expected tools found: {len(found_expected)}/{len(expected)}")
            print(f"   ⚡ Capabilities: {', '.join(found_expected)}")
            
        else:
            print(f"   ❌ Planning failed: {result.get('error')}")
    
    print("\n🏆 FLUX KONTEXT MAX CAPABILITIES:")
    print("   🎨 Style Transfer: Watercolor, oil painting, sketches, digital art")
    print("   👔 Object Changes: Hair, clothing, accessories, colors")
    print("   📝 Text Editing: Signs, posters, labels, logos")
    print("   🌆 Background Swap: Indoor, outdoor, studio, fantasy environments")
    print("   👤 Character Consistency: Identity preservation across variations")


async def demonstrate_tool_chaining():
    """Show complex tool chaining with Flux editing."""
    print("\n\n🔗 ADVANCED TOOL CHAINING DEMONSTRATION")
    print("=" * 60)
    
    agent = ToolFirstAgent()
    
    # Complex multi-step editing request
    complex_request = "Transform this corporate headshot: change to watercolor style, make the hair blonde, replace background with a tropical beach, and ensure the final result maintains professional quality"
    
    print(f"Complex Request: {complex_request}")
    print("\nThis should demonstrate advanced tool chaining:")
    print("1. Style Transfer → Watercolor conversion")
    print("2. Object Change → Hair color modification")
    print("3. Background Swap → Tropical beach environment")
    print("4. Quality Assessment → Professional validation")
    
    result = await agent.process_request(complex_request)
    
    if result["success"]:
        workflow = result["workflow_plan"]
        print(f"\n📋 Actual Planned Workflow ({len(workflow['workflow_plan'])} steps):")
        
        total_estimated_time = 0
        for step in workflow['workflow_plan']:
            tool_name = step['tool_name']
            is_flux = any(x in tool_name.lower() for x in ["flux", "style", "object", "text", "background", "character"])
            prefix = "🎨" if is_flux else "🔧"
            
            print(f"   {step['step']}. {prefix} {tool_name}")
            print(f"      └─ {step['description']}")
            print(f"      └─ Expected: {step['expected_output']}")
            
            # Estimate processing time for Flux operations
            if is_flux:
                estimated_time = 30  # Flux operations typically take 15-45 seconds
                total_estimated_time += estimated_time
                print(f"      └─ Est. Time: {estimated_time}s (Flux Kontext Max)")
        
        print(f"\n⏱️  Total Estimated Processing Time: {total_estimated_time}s")
        print(f"💡 Reasoning: {workflow.get('reasoning', 'Multi-step professional editing workflow')}")
        print(f"🎯 Confidence: {workflow.get('confidence', 'High')} (Professional tool chain)")
        
    else:
        print(f"❌ Planning failed: {result.get('error')}")


async def show_flux_extensibility():
    """Demonstrate how easy it is to add new Flux capabilities."""
    print("\n\n🔄 FLUX EXTENSIBILITY DEMONSTRATION") 
    print("=" * 60)
    
    print("Adding new Flux capabilities is as simple as:")
    print("""
    # 1. Create specialized Flux tool
    class NewFluxCapabilityTool(Tool):
        def __init__(self, config):
            super().__init__(
                name="new_flux_capability",
                description="Advanced Flux feature for specific editing",
                category=ToolCategory.IMAGE_EDITING,
                input_schema={...},
                output_schema={...}
            )
            self.flux_tool = FluxKontextMaxTool(config)
        
        async def invoke(self, input_data):
            # Prepare specialized Flux input
            flux_input = {
                "operation_type": "new_capability",
                ...
            }
            return await self.flux_tool.invoke(flux_input)
    
    # 2. Register tool
    tool_registry.register(NewFluxCapabilityTool())
    
    # 3. Agent automatically uses it in workflows!
    """)
    
    print("🎯 Current Flux Integrations:")
    print("   ✅ FluxKontextMaxTool - Unified advanced editing")
    print("   ✅ StyleTransferTool - Art style conversion")
    print("   ✅ ObjectChangeTool - Selective modifications")
    print("   ✅ TextEditingTool - Text replacement")
    print("   ✅ BackgroundSwapTool - Environment changes")
    print("   ✅ CharacterConsistencyTool - Identity preservation")
    
    print("\n🔮 Future Flux Capabilities (easy to add):")
    print("   🔜 Video Style Transfer")
    print("   🔜 3D Model Generation")
    print("   🔜 Advanced Inpainting")
    print("   🔜 Pose Manipulation")
    print("   🔜 Lighting Adjustment")


async def compare_approaches():
    """Side-by-side comparison of both approaches."""
    print("\n\n📊 DETAILED COMPARISON")
    print("=" * 60)
    
    comparison = [
        ("Flexibility", "Fixed workflow", "Dynamic planning"),
        ("Extensibility", "Code changes needed", "Drop-in tools"),
        ("Request Handling", "Limited patterns", "Any request type"),
        ("Tool Discovery", "Hardcoded", "Automatic"),
        ("Workflow Adaptation", "None", "AI-powered"),
        ("Error Handling", "Pipeline breaks", "Graceful degradation"),
        ("Maintainability", "Coupled code", "Modular tools"),
        ("Testing", "Integration only", "Unit + Integration"),
        ("Image Editing", "Basic generation", "Professional editing"),
        ("Style Transfer", "Not available", "Multiple art styles"),
        ("Text Editing", "Not available", "Advanced typography"),
        ("Background Swap", "Not available", "Seamless environments"),
        ("Character Work", "Not available", "Identity consistency"),
    ]
    
    print(f"{'Aspect':<20} {'Old Approach':<20} {'Tool-First + Flux':<25}")
    print("-" * 67)
    
    for aspect, old, new in comparison:
        print(f"{aspect:<20} {old:<20} {new:<25}")
    
    print("\n🎯 CONCLUSION:")
    print("Tool-first architecture with Flux Kontext Max provides professional-grade")
    print("image editing capabilities that transform Pic Arcade into a cutting-edge")
    print("AI platform for creative professionals and content creators.")


async def main():
    """Run the complete demonstration."""
    print("🚀 PIC ARCADE: TOOL-FIRST ARCHITECTURE WITH FLUX KONTEXT MAX")
    print("Advanced professional image editing with dynamic tool selection")
    print("=" * 70)
    
    # Check API keys with updated requirements
    required_keys = ["OPENAI_API_KEY", "PERPLEXITY_API_KEY"]
    optional_keys = ["REPLICATE_API_TOKEN"]
    
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing required API keys: {', '.join(missing_keys)}")
        print("Please set the required environment variables in your .env file:")
        for key in missing_keys:
            print(f"   {key}=your_api_key_here")
        print(f"\nOptional (for Flux Kontext Max): {', '.join(optional_keys)}")
        return
    
    print("✅ Required API keys found - proceeding with demonstration")
    
    # Show which keys are available
    available_keys = []
    if os.getenv("OPENAI_API_KEY"):
        available_keys.append("OpenAI (GPT-4o)")
    if os.getenv("PERPLEXITY_API_KEY"):
        available_keys.append("Perplexity (Search)")
    if os.getenv("REPLICATE_API_TOKEN"):
        available_keys.append("Replicate (Flux Kontext Max)")
    
    print(f"🔑 Available APIs: {', '.join(available_keys)}")
    
    if os.getenv("REPLICATE_API_TOKEN"):
        print("🎨 Flux Kontext Max capabilities enabled!")
    else:
        print("⚠️  Flux Kontext Max disabled (no Replicate token)")
        print("   Demo will show planning only, not actual image generation")
    
    try:
        # Run demonstrations
        await demo_old_approach()
        await demo_tool_first_approach()
        await demonstrate_flux_capabilities()
        await demonstrate_tool_chaining()
        await show_flux_extensibility()
        await compare_approaches()
        
        print("\n\n🎉 DEMONSTRATION COMPLETE!")
        print("The tool-first architecture with Flux Kontext Max transforms")
        print("Pic Arcade into a professional-grade AI image editing platform.")
        
        if os.getenv("REPLICATE_API_TOKEN"):
            print("\n🚀 Try the full Flux demo:")
            print("   python packages/agentic/examples/flux_kontext_demo.py")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 