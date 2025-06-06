#!/usr/bin/env python3
"""
Debug Frontend Issue

Quick script to debug why the frontend is still showing the old workflow
instead of using the context-aware planning.
"""

import sys
from pathlib import Path

# Add the agentic package to the path
sys.path.append(str(Path(__file__).parent / "src"))

def check_conversation_context():
    """Check if conversation context is working."""
    print("🔍 CHECKING CONVERSATION CONTEXT")
    print("=" * 40)
    
    try:
        from src.pic_arcade_agentic.utils.conversation_context import conversation_context
        
        # Check current state
        total_results = len(conversation_context.generation_results)
        recent_images = len(conversation_context.get_recent_images())
        
        print(f"📊 Current State:")
        print(f"   Total results: {total_results}")
        print(f"   Recent images: {recent_images}")
        
        if total_results == 0:
            print("❌ No conversation history found!")
            print("💡 This means your first 'create cat' request didn't store results")
            
            # Simulate adding cat image
            print("\n🧪 Simulating cat generation storage...")
            conversation_context.add_generation_result(
                prompt="Create an image of a cat",
                intent="generate_image",
                result_type="image",
                result_data={"image_url": "https://example.com/cat.png"},
                agent_name="ToolFirstAgent",
                request_id="debug_cat_001"
            )
            
            print("✅ Added cat image to context")
            print(f"   Now has {len(conversation_context.generation_results)} results")
        
        # Test edit detection
        print(f"\n🎯 Testing edit detection for 'Add a hat':")
        edit_context = conversation_context.detect_edit_context("Add a hat", "edit_image")
        
        print(f"   Is edit: {edit_context['is_edit']}")
        print(f"   Edit type: {edit_context['edit_type']}")
        print(f"   Has target image: {edit_context['target_image'] is not None}")
        print(f"   Confidence: {edit_context['confidence']:.2f}")
        
        if edit_context['is_edit'] and edit_context['target_image']:
            print("✅ Context detection working correctly!")
            return True
        else:
            print("❌ Context detection not working properly")
            return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the right directory")
        return False

def check_workflow_planner():
    """Check if workflow planner is enhanced."""
    print(f"\n🛠️ CHECKING WORKFLOW PLANNER")
    print("=" * 35)
    
    try:
        from src.pic_arcade_agentic.tools.workflow_tools import WorkflowPlanningTool
        
        # Check if it has conversation_context in input schema
        planner = WorkflowPlanningTool({"openai_api_key": "test"})
        input_schema = planner.input_schema
        
        has_context_input = "conversation_context" in input_schema.get("properties", {})
        
        print(f"📋 Input Schema Analysis:")
        print(f"   Has conversation_context: {has_context_input}")
        
        if has_context_input:
            print("✅ Workflow planner is enhanced!")
            
            # Check for _ensure_edit_compliance method
            has_compliance_method = hasattr(planner, '_ensure_edit_compliance')
            print(f"   Has edit compliance: {has_compliance_method}")
            
            return True
        else:
            print("❌ Workflow planner is NOT enhanced")
            print("💡 The enhanced code may not be loaded")
            return False
        
    except Exception as e:
        print(f"❌ Error checking workflow planner: {e}")
        return False

def check_tool_agent():
    """Check if ToolFirstAgent is enhanced."""
    print(f"\n🤖 CHECKING TOOL FIRST AGENT")
    print("=" * 30)
    
    try:
        from src.pic_arcade_agentic.agents.tool_agent import ToolFirstAgent
        
        # Check if it has context analysis method
        agent = ToolFirstAgent()
        has_context_method = hasattr(agent, '_analyze_conversation_context')
        has_store_method = hasattr(agent, '_store_generation_result')
        
        print(f"📋 Agent Method Analysis:")
        print(f"   Has context analysis: {has_context_method}")
        print(f"   Has result storage: {has_store_method}")
        
        if has_context_method and has_store_method:
            print("✅ ToolFirstAgent is enhanced!")
            return True
        else:
            print("❌ ToolFirstAgent is NOT enhanced")
            print("💡 You may be using the old version")
            return False
        
    except Exception as e:
        print(f"❌ Error checking ToolFirstAgent: {e}")
        return False

def diagnose_frontend_issue():
    """Diagnose why frontend shows old workflow."""
    print(f"\n🌐 DIAGNOSING FRONTEND ISSUE")
    print("=" * 35)
    
    print("❓ Questions to check:")
    print("1. Are you using the enhanced code in your API?")
    print("2. Is conversation context being populated after cat generation?")
    print("3. Is the API calling the enhanced ToolFirstAgent?")
    print("4. Are there any caching issues?")
    
    print(f"\n🔧 Quick fixes to try:")
    print("1. Restart your API server")
    print("2. Clear any caches")
    print("3. Check your API endpoint is using the enhanced agent")
    print("4. Verify the workflow planner is getting conversation_context")
    
    print(f"\n📝 Debug steps:")
    print("1. Check API logs for context analysis messages")
    print("2. Look for decision logs from enhanced agent")
    print("3. Verify conversation_context has recent images")
    print("4. Test the workflow planner directly with context")

def show_expected_vs_actual():
    """Show what should happen vs what's happening."""
    print(f"\n📊 EXPECTED VS ACTUAL BEHAVIOR")
    print("=" * 40)
    
    print("✅ EXPECTED (with fixes):")
    print("   1. User: 'Create a cat'")
    print("      → Agent generates cat image")
    print("      → Image stored in conversation_context")
    print("      → Returns cat image URL")
    print("")
    print("   2. User: 'Add a hat'")
    print("      → Agent detects edit intent")
    print("      → Finds cat image in context")
    print("      → Plans workflow: prompt_parser → flux_kontext_max")
    print("      → flux_kontext_max gets original cat image + 'add hat' prompt")
    print("      → Returns edited cat with hat")
    
    print("\n❌ ACTUAL (from your output):")
    print("   1. User: 'Create a cat' (?)")
    print("      → Generated cat but context not stored")
    print("")
    print("   2. User: 'Add a hat'")
    print("      → Agent detects edit intent ✅")
    print("      → NO original image found ❌")
    print("      → Plans workflow: prompt_parser → object_change ❌")
    print("      → object_change works without original image")
    print("      → Returns new image of 'guy in hat' ❌")

def main():
    """Run all diagnostic checks."""
    print("🚨 FRONTEND ISSUE DIAGNOSTIC")
    print("=" * 40)
    print("Checking why 'Add a hat' still uses old workflow...")
    
    # Run all checks
    context_ok = check_conversation_context()
    planner_ok = check_workflow_planner()
    agent_ok = check_tool_agent()
    
    diagnose_frontend_issue()
    show_expected_vs_actual()
    
    print(f"\n🎯 DIAGNOSTIC SUMMARY")
    print("=" * 25)
    print(f"Conversation Context: {'✅' if context_ok else '❌'}")
    print(f"Workflow Planner: {'✅' if planner_ok else '❌'}")
    print(f"Tool Agent: {'✅' if agent_ok else '❌'}")
    
    if all([context_ok, planner_ok, agent_ok]):
        print("\n🎉 All components are enhanced!")
        print("💡 Issue is likely in frontend/API integration")
        print("📞 Check if your API is actually using the enhanced code")
    else:
        print("\n⚠️ Some components need fixing")
        print("💡 Make sure all enhanced code is properly loaded")
    
    print(f"\n🚀 NEXT STEPS:")
    print("1. Run this diagnostic script")
    print("2. Fix any ❌ issues found")
    print("3. Test 'Create a cat' → 'Add a hat' again")
    print("4. Check decision logs for context usage")

if __name__ == "__main__":
    main() 