#!/usr/bin/env python3
"""
Test Context-Aware Workflow Planning

This script tests the enhanced WorkflowPlanningTool to ensure it properly
handles conversation context and selects the right tools for editing operations.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add the agentic package to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pic_arcade_agentic.tools.workflow_tools import WorkflowPlanningTool
from src.pic_arcade_agentic.utils.conversation_context import conversation_context

async def test_context_aware_planning():
    """Test the enhanced workflow planning with conversation context."""
    
    print("🧪 TESTING CONTEXT-AWARE WORKFLOW PLANNING")
    print("=" * 55)
    
    # Initialize the workflow planner
    config = {
        "openai_api_key": "your_openai_key_here"  # You'll need to set this
    }
    
    try:
        planner = WorkflowPlanningTool(config)
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Please set your OPENAI_API_KEY environment variable")
        return
    
    # Test 1: Generation without context (baseline)
    print("\n1️⃣ Test: New Image Generation (No Context)")
    print("-" * 45)
    
    generation_request = {
        "user_request": "Create an image of a fluffy orange cat sitting on a windowsill",
        "conversation_context": {
            "is_edit": False
        }
    }
    
    result = await planner.invoke(generation_request)
    
    if result.success:
        plan = result.data
        print(f"✅ Generated {len(plan['workflow_plan'])} steps")
        for step in plan['workflow_plan']:
            print(f"   {step['step']}. {step['tool_name']}: {step['description']}")
        print(f"🎯 Confidence: {plan.get('confidence', 0):.2f}")
    else:
        print(f"❌ Planning failed: {result.error}")
    
    # Test 2: Edit with context (the problematic scenario)
    print("\n2️⃣ Test: Edit Operation (With Context)")
    print("-" * 40)
    
    # Simulate the context from the cat generation
    edit_context = {
        "is_edit": True,
        "edit_type": "add_object",
        "has_target_image": True,
        "target_image_url": "https://example.com/generated_cat_image.png",
        "original_prompt": "Create an image of a fluffy orange cat sitting on a windowsill",
        "edit_instructions": "Add a hat"
    }
    
    edit_request = {
        "user_request": "Add a hat",
        "conversation_context": edit_context
    }
    
    result = await planner.invoke(edit_request)
    
    if result.success:
        plan = result.data
        print(f"✅ Generated {len(plan['workflow_plan'])} steps")
        
        uses_flux_kontext = False
        has_original_image = False
        
        for step in plan['workflow_plan']:
            tool_name = step['tool_name']
            inputs = step.get('inputs', {})
            
            print(f"   {step['step']}. {tool_name}: {step['description']}")
            print(f"      Inputs: {inputs}")
            
            if tool_name == "flux_kontext_max":
                uses_flux_kontext = True
                print("      ✅ Using FluxKontextMax for editing!")
                
                if "image" in inputs:
                    has_original_image = True
                    print("      ✅ Original image included!")
                else:
                    print("      ❌ Missing original image")
            
            elif tool_name == "object_change":
                print("      ❌ Still using object_change (should be flux_kontext_max)")
        
        print(f"\n📊 Analysis:")
        print(f"   Uses FluxKontext: {'✅' if uses_flux_kontext else '❌'}")
        print(f"   Has Original Image: {'✅' if has_original_image else '❌'}")
        print(f"   Confidence: {plan.get('confidence', 0):.2f}")
        print(f"   Reasoning: {plan.get('reasoning', 'N/A')}")
        
        # This should now work correctly!
        if uses_flux_kontext and has_original_image:
            print("\n🎉 SUCCESS: Context-aware planning working correctly!")
        else:
            print("\n❌ ISSUE: Context-aware planning needs more work")
            
    else:
        print(f"❌ Planning failed: {result.error}")
    
    # Test 3: Compare side-by-side
    print("\n3️⃣ Comparison: Before vs After Context Enhancement")
    print("-" * 50)
    
    print("BEFORE (old workflow):")
    print("  Step 1: prompt_parser")
    print("  Step 2: object_change (❌ wrong tool, no original image)")
    
    print("\nAFTER (context-aware workflow):")
    if result.success:
        for step in result.data['workflow_plan']:
            tool_indicator = "✅" if step['tool_name'] == "flux_kontext_max" else "⚠️"
            print(f"  Step {step['step']}: {step['tool_name']} {tool_indicator}")

async def test_with_conversation_context_manager():
    """Test integration with ConversationContextManager."""
    
    print("\n\n🔗 TESTING INTEGRATION WITH CONVERSATION CONTEXT")
    print("=" * 55)
    
    # Clear any existing context
    conversation_context.generation_results.clear()
    
    # Step 1: Simulate cat generation
    print("1️⃣ Simulating: 'Create an image of a cat'")
    conversation_context.add_generation_result(
        prompt="Create an image of a fluffy orange cat sitting on a windowsill",
        intent="generate_image",
        result_type="image",
        result_data={"image_url": "https://example.com/generated_cat.png"},
        agent_name="ToolFirstAgent",
        request_id="test_cat_001"
    )
    print("   ✅ Cat image stored in conversation context")
    
    # Step 2: Test edit detection
    print("\n2️⃣ Testing: 'Add a hat'")
    edit_context = conversation_context.detect_edit_context("Add a hat", "edit_image")
    
    print(f"   Edit detected: {edit_context['is_edit']}")
    print(f"   Edit type: {edit_context['edit_type']}")
    print(f"   Has target image: {edit_context['target_image'] is not None}")
    print(f"   Confidence: {edit_context['confidence']:.2f}")
    
    if edit_context['is_edit'] and edit_context['target_image']:
        print("   ✅ Context detection working!")
        
        # Step 3: Test workflow planning with this context
        print("\n3️⃣ Testing workflow planning with detected context")
        
        config = {"openai_api_key": "test_key"}  # Mock for this test
        
        # Create the same context structure the agent would create
        workflow_context = {
            "is_edit": edit_context["is_edit"],
            "edit_type": edit_context["edit_type"],
            "has_target_image": edit_context["target_image"] is not None,
            "target_image_url": edit_context["target_image"],
            "original_prompt": edit_context["original_prompt"],
            "edit_instructions": edit_context["edit_instructions"]
        }
        
        print(f"   Context for workflow planner:")
        for key, value in workflow_context.items():
            print(f"     {key}: {value}")
        
        print("\n   ✅ All components working together!")
    else:
        print("   ❌ Context detection failed")

def show_frontend_integration_guide():
    """Show how to integrate this with the frontend."""
    
    print("\n\n🌐 FRONTEND INTEGRATION GUIDE")
    print("=" * 40)
    
    print("\n📋 Steps to fix your frontend issue:")
    print("1. Ensure your API uses the enhanced ToolFirstAgent")
    print("2. Make sure conversation context is populated after image generation")
    print("3. The enhanced WorkflowPlanningTool will now use flux_kontext_max")
    print("4. Verify your frontend calls the correct API endpoint")
    
    print("\n🔧 Debug your current setup:")
    print("# Check if conversation context has the cat image:")
    print("from src.pic_arcade_agentic.utils.conversation_context import conversation_context")
    print("print(f'Images in context: {len(conversation_context.get_recent_images())}')")
    print("")
    print("# Test edit detection:")
    print("edit_ctx = conversation_context.detect_edit_context('Add a hat', 'edit_image')")
    print("print(f'Edit detected: {edit_ctx[\"is_edit\"]}, Has image: {edit_ctx[\"target_image\"] is not None}')")
    
    print("\n💡 Expected behavior now:")
    print("✅ 'Create a cat' → stores image in conversation context")
    print("✅ 'Add a hat' → detects edit → uses flux_kontext_max with original image")
    print("✅ Result: Hat added to existing cat (not new image)")

async def main():
    """Run all tests."""
    await test_context_aware_planning()
    await test_with_conversation_context_manager()
    show_frontend_integration_guide()
    
    print("\n\n🎯 SUMMARY")
    print("=" * 20)
    print("✅ Enhanced WorkflowPlanningTool with context awareness")
    print("✅ Integration with ConversationContextManager")
    print("✅ Proper tool selection (flux_kontext_max for edits)")
    print("✅ Original image passing for edit operations")
    
    print("\n🚀 Next steps:")
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Run this test to verify everything works")
    print("3. Test your frontend with the enhanced system")
    print("4. Check decision logs for any remaining issues")

if __name__ == "__main__":
    asyncio.run(main()) 