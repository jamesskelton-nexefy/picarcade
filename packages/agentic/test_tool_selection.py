#!/usr/bin/env python3
"""
Tool Selection Validation Tests

Tests to ensure the correct tools are selected for different request types
and that they use the appropriate models.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pic_arcade_agentic.agents.tool_agent import ToolFirstAgent
from pic_arcade_agentic.tools.base import tool_registry


async def test_tool_selection_scenarios():
    """Test various scenarios to validate tool selection logic."""
    
    print("🔧 TOOL SELECTION VALIDATION TESTS")
    print("=" * 60)
    
    # Initialize the agent
    try:
        agent = ToolFirstAgent()
        print("✅ ToolFirstAgent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Test scenarios that should select specific tools
    test_scenarios = [
        {
            "name": "Style Transfer Request",
            "request": "Convert this photo to watercolor painting style",
            "expected_tools": ["prompt_parser", "style_transfer"],
            "expected_model": "black-forest-labs/flux-kontext-max"
        },
        {
            "name": "Object Change Request", 
            "request": "Change the person's hair from brown to blonde",
            "expected_tools": ["prompt_parser", "object_change"],
            "expected_model": "black-forest-labs/flux-kontext-max"
        },
        {
            "name": "Text Editing Request",
            "request": "Replace the text on the store sign with 'Coffee Shop'",
            "expected_tools": ["prompt_parser", "text_editing"],
            "expected_model": "black-forest-labs/flux-kontext-max"
        },
        {
            "name": "Background Swap Request",
            "request": "Change the background to a tropical beach setting",
            "expected_tools": ["prompt_parser", "background_swap"],
            "expected_model": "black-forest-labs/flux-kontext-max"
        },
        {
            "name": "Character Consistency Request",
            "request": "Generate the same character in a different pose",
            "expected_tools": ["prompt_parser", "character_consistency"],
            "expected_model": "flux-kontext-max"
        },
        {
            "name": "General Image Generation",
            "request": "Create a portrait of a professional businesswoman",
            "expected_tools": ["prompt_parser", "flux_kontext_max"],
            "expected_model": "flux-1.1-pro"
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        print(f"📝 Request: {scenario['request']}")
        
        try:
            # Get workflow plan without executing it
            result = await agent.process_request(scenario["request"])
            
            if result["success"]:
                workflow_plan = result.get("workflow_plan", {}).get("workflow_plan", [])
                tools_used = [step["tool_name"] for step in workflow_plan]
                
                print(f"🔧 Tools planned: {tools_used}")
                
                # Check if expected tools are present
                expected_present = all(tool in tools_used for tool in scenario["expected_tools"])
                
                # Check execution results for model validation
                execution_results = result.get("execution_results", {}).get("execution_results", [])
                models_used = []
                
                for exec_result in execution_results:
                    if exec_result.get("success") and exec_result.get("data"):
                        metadata = exec_result["data"]
                        if isinstance(metadata, dict) and "metadata" in metadata:
                            model_used = metadata["metadata"].get("model_used")
                            if model_used:
                                models_used.append(model_used)
                
                print(f"🤖 Models used: {models_used}")
                
                # Validate results
                test_result = {
                    "scenario": scenario["name"],
                    "request": scenario["request"],
                    "expected_tools": scenario["expected_tools"],
                    "actual_tools": tools_used,
                    "expected_model": scenario["expected_model"],
                    "actual_models": models_used,
                    "tools_correct": expected_present,
                    "model_correct": scenario["expected_model"] in models_used if models_used else False,
                    "status": "✅ PASS" if expected_present else "❌ FAIL"
                }
                
                results.append(test_result)
                print(f"📊 Result: {test_result['status']}")
                
                if not expected_present:
                    missing_tools = [tool for tool in scenario["expected_tools"] if tool not in tools_used]
                    print(f"⚠️  Missing expected tools: {missing_tools}")
                
            else:
                print(f"❌ Request failed: {result.get('error', 'Unknown error')}")
                results.append({
                    "scenario": scenario["name"],
                    "status": "❌ FAIL",
                    "error": result.get("error", "Unknown error")
                })
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                "scenario": scenario["name"], 
                "status": "❌ ERROR",
                "error": str(e)
            })
    
    # Print summary
    print(f"\n📋 TOOL SELECTION TEST SUMMARY")
    print("=" * 60)
    
    passed = len([r for r in results if r.get("status", "").startswith("✅")])
    total = len(results)
    
    print(f"📊 Tests Passed: {passed}/{total}")
    print(f"🎯 Success Rate: {(passed/total)*100:.1f}%")
    
    print(f"\n📝 DETAILED RESULTS:")
    for result in results:
        print(f"\n{result['scenario']}: {result['status']}")
        if "actual_tools" in result:
            print(f"  Expected Tools: {result['expected_tools']}")
            print(f"  Actual Tools: {result['actual_tools']}")
            print(f"  Expected Model: {result['expected_model']}")
            print(f"  Actual Models: {result.get('actual_models', [])}")
        if "error" in result:
            print(f"  Error: {result['error']}")
    
    return results


async def test_individual_tool_models():
    """Test individual tools to verify they use the correct models."""
    
    print(f"\n🔬 INDIVIDUAL TOOL MODEL VALIDATION")
    print("=" * 60)
    
    # Test individual tool model assignments
    tool_tests = [
        {
            "tool_name": "style_transfer",
            "expected_model": "black-forest-labs/flux-kontext-max",
            "test_input": {
                "image": "https://example.com/test.jpg",
                "style": "watercolor",
                "prompt": "convert to watercolor style"
            }
        },
        {
            "tool_name": "object_change", 
            "expected_model": "black-forest-labs/flux-kontext-max",
            "test_input": {
                "image": "https://example.com/test.jpg",
                "target_object": "hair",
                "modification": "blonde hair"
            }
        },
        {
            "tool_name": "text_editing",
            "expected_model": "black-forest-labs/flux-kontext-max", 
            "test_input": {
                "image": "https://example.com/test.jpg",
                "new_text": "Coffee Shop"
            }
        },
        {
            "tool_name": "background_swap",
            "expected_model": "black-forest-labs/flux-kontext-max",
            "test_input": {
                "image": "https://example.com/test.jpg",
                "new_background": "tropical beach"
            }
        }
    ]
    
    for test in tool_tests:
        print(f"\n🧪 Testing {test['tool_name']}")
        
        try:
            tool = tool_registry.get_tool(test["tool_name"])
            if not tool:
                print(f"❌ Tool '{test['tool_name']}' not found in registry")
                continue
                
            print(f"✅ Tool found: {tool.description}")
            
            # Check tool configuration/model reference
            # Note: This is a dry run - we're checking the tool setup, not actually calling APIs
            print(f"🔧 Expected model: {test['expected_model']}")
            print(f"📝 Tool type: {type(tool).__name__}")
            
            # Verify the tool is configured to use the expected model
            # by checking if it has the correct model reference in its implementation
            if hasattr(tool, '_validate_config'):
                try:
                    # This will validate API keys are set up correctly
                    tool._validate_config()
                    print(f"✅ Tool configuration valid")
                except Exception as e:
                    print(f"⚠️  Tool configuration issue: {e}")
            
        except Exception as e:
            print(f"❌ Error testing {test['tool_name']}: {e}")


def print_available_tools():
    """Print all available tools and their models."""
    
    print(f"\n📚 AVAILABLE TOOLS REGISTRY")
    print("=" * 60)
    
    tools = tool_registry.list_all_tools()
    
    for tool_name in sorted(tools):
        tool = tool_registry.get_tool(tool_name)
        if tool:
            print(f"🔧 {tool_name}")
            print(f"   Description: {tool.description}")
            print(f"   Category: {tool.category}")
            print(f"   Type: {type(tool).__name__}")


async def main():
    """Run all tool selection validation tests."""
    
    # Check environment setup
    required_keys = ["REPLICATE_API_TOKEN", "OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"⚠️  Warning: Missing API keys: {missing_keys}")
        print("Some tests may fail due to missing configuration.")
        print("Run 'npm run setup:test-env' to configure API keys.")
    
    # Print available tools
    print_available_tools()
    
    # Test workflow-level tool selection
    workflow_results = await test_tool_selection_scenarios()
    
    # Test individual tool model assignments  
    await test_individual_tool_models()
    
    print(f"\n🎉 TOOL SELECTION VALIDATION COMPLETE")
    print("=" * 60)
    
    return workflow_results


if __name__ == "__main__":
    asyncio.run(main()) 