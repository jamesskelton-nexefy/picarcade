#!/usr/bin/env python3
"""
Debug Script for Context Issue Analysis

This script analyzes the decision logs to understand why the agent
failed to maintain context between image generation and editing.
"""

import json
from pathlib import Path
from src.pic_arcade_agentic.utils.decision_logger import decision_logger, DecisionType

def analyze_context_failure():
    """
    Analyze the decision logs to understand the context failure.
    """
    print("🔍 ANALYZING CONTEXT FAILURE")
    print("=" * 50)
    
    # Get decision history
    history = decision_logger.get_decision_history()
    
    if not history:
        print("❌ No decision history found. Please run your prompts first.")
        return
    
    # Find the "Add a hat" decision
    edit_decisions = []
    for decision in history:
        if decision.agent_name == "ToolFirstAgent":
            # Check if any step mentions editing or hat
            for step in decision.steps:
                if ('hat' in str(step.input_data).lower() or 
                    'edit' in str(step.output_data).lower()):
                    edit_decisions.append(decision)
                    break
    
    if not edit_decisions:
        print("❌ No edit-related decisions found in logs.")
        return
    
    print(f"📊 Found {len(edit_decisions)} edit-related decisions")
    
    # Analyze the most recent edit decision
    latest_edit = edit_decisions[-1]
    print(f"\n🔸 Analyzing Request: {latest_edit.request_id}")
    print(f"   Agent: {latest_edit.agent_name}")
    print(f"   Success: {'✅' if latest_edit.success else '❌'}")
    print(f"   Total Steps: {latest_edit.total_steps}")
    
    print(f"\n📝 Decision Steps Analysis:")
    
    context_maintained = False
    original_image_referenced = False
    flux_kontext_used = False
    
    for i, step in enumerate(latest_edit.steps):
        print(f"\n   Step {i+1}: {step.decision_type.value}")
        print(f"   Reasoning: {step.decision_reasoning}")
        
        # Check for context awareness
        input_str = str(step.input_data).lower()
        output_str = str(step.output_data).lower()
        
        if ('previous' in input_str or 'original' in input_str or 
            'existing' in input_str or 'context' in input_str):
            context_maintained = True
            print(f"   ✅ Context awareness detected")
        
        if ('image' in input_str and ('url' in input_str or 'path' in input_str)):
            original_image_referenced = True
            print(f"   ✅ Original image referenced")
        
        if ('flux' in output_str and 'kontext' in output_str):
            flux_kontext_used = True
            print(f"   ✅ Flux Kontext model selected")
        
        if step.decision_type == DecisionType.TOOL_SELECTION:
            tools_mentioned = step.output_data.get('selected_tool', '') or step.output_data.get('tools_required', [])
            print(f"   🔧 Tools selected: {tools_mentioned}")
        
        if step.confidence_score:
            confidence_icon = "🟢" if step.confidence_score > 0.7 else "🟡" if step.confidence_score > 0.4 else "🔴"
            print(f"   {confidence_icon} Confidence: {step.confidence_score:.2f}")
    
    print(f"\n📋 Context Failure Analysis:")
    print(f"   Context Maintained: {'✅ Yes' if context_maintained else '❌ No'}")
    print(f"   Original Image Referenced: {'✅ Yes' if original_image_referenced else '❌ No'}")
    print(f"   Flux Kontext Used: {'✅ Yes' if flux_kontext_used else '❌ No'}")
    
    # Identify the root cause
    print(f"\n🎯 Root Cause Analysis:")
    
    if not context_maintained:
        print(f"   🚨 ISSUE: No context awareness in decision-making")
        print(f"      The agent didn't recognize this as an edit of existing content")
    
    if not original_image_referenced:
        print(f"   🚨 ISSUE: Original image not provided to tools")
        print(f"      The editing tool received no reference to the cat image")
    
    if not flux_kontext_used:
        print(f"   🚨 ISSUE: Wrong tool selected for image editing")
        print(f"      Should use FluxKontextMaxTool for image-to-image editing")
    
    return {
        'context_maintained': context_maintained,
        'original_image_referenced': original_image_referenced,
        'flux_kontext_used': flux_kontext_used,
        'decision': latest_edit
    }

def suggest_fixes():
    """
    Suggest fixes for the context issue.
    """
    print(f"\n💡 SUGGESTED FIXES")
    print("=" * 30)
    
    print(f"\n1️⃣ Add Context Memory to Agent")
    print(f"   • Store previous generation results in agent memory")
    print(f"   • Include conversation history in decision-making")
    print(f"   • Add 'previous_outputs' to workflow planning input")
    
    print(f"\n2️⃣ Improve Workflow Planning Logic")
    print(f"   • Detect edit intentions more accurately")
    print(f"   • When intent='edit_image', check for previous image context")
    print(f"   • Route to appropriate image-to-image tools (FluxKontext)")
    
    print(f"\n3️⃣ Enhanced Tool Selection")
    print(f"   • Use FluxKontextMaxTool for image editing with reference")
    print(f"   • Pass original image URL to editing tools")
    print(f"   • Validate image inputs before tool execution")
    
    print(f"\n4️⃣ Decision Logging Improvements")
    print(f"   • Log context checking decisions explicitly")
    print(f"   • Track image reference passing between tools")
    print(f"   • Add metadata for multi-turn conversation handling")

def create_context_aware_example():
    """
    Show how the decision logging would look with proper context handling.
    """
    print(f"\n📖 EXAMPLE: Proper Context-Aware Decision Log")
    print("=" * 50)
    
    example_log = {
        "step_id": "context_check_001",
        "agent_name": "ToolFirstAgent", 
        "decision_type": "workflow_planning",
        "decision_reasoning": "User prompt 'Add a hat' detected as edit intent. Checking conversation history for previous image generation.",
        "input_data": {
            "current_prompt": "Add a hat",
            "conversation_history": [
                {"prompt": "Create an image of a cat", "result": "cat_image_url.png"}
            ],
            "intent": "edit_image"
        },
        "output_data": {
            "context_found": True,
            "original_image": "cat_image_url.png", 
            "edit_type": "object_addition",
            "selected_tool": "flux_kontext_max",
            "tool_inputs": {
                "original_image": "cat_image_url.png",
                "edit_prompt": "Add a hat to the cat"
            }
        },
        "confidence_score": 0.95,
        "metadata": {
            "context_aware": True,
            "multi_turn_handling": True,
            "image_to_image_editing": True
        }
    }
    
    print(json.dumps(example_log, indent=2))

if __name__ == "__main__":
    # Run the analysis
    result = analyze_context_failure()
    
    # Suggest fixes
    suggest_fixes()
    
    # Show proper example
    create_context_aware_example()
    
    print(f"\n🎉 Analysis complete! Use these insights to improve context handling.") 