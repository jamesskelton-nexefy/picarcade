#!/usr/bin/env python3
"""
Decision Logging Demonstration Script

This script demonstrates the comprehensive decision logging system
implemented across all agents in the PicArcade agentic package.

Run this script to see detailed decision tracking in action.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

# Import the decision logger and agents
from src.pic_arcade_agentic.utils.decision_logger import decision_logger, DecisionType
from src.pic_arcade_agentic.workflow.orchestrator import WorkflowOrchestrator
from src.pic_arcade_agentic.agents.tool_agent import ToolFirstAgent
from src.pic_arcade_agentic.agents.prompt_parser import PromptParsingAgent
from src.pic_arcade_agentic.agents.reference_retriever import ReferenceRetrievalAgent


async def demonstrate_decision_logging():
    """
    Demonstrate the decision logging system with various agent operations.
    """
    print("🔍 PICARCADE AGENT DECISION LOGGING DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Example prompts to test with
    test_prompts = [
        "Create a portrait of Emma Stone in Renaissance style",
        "Generate a landscape with Van Gogh's painting style",
        "Put me in Taylor Swift's red dress from the awards show",
        "Create a cyberpunk cityscape with neon lighting"
    ]
    
    print("📋 Test Prompts:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  {i}. {prompt}")
    print()
    
    # Demonstrate different agents
    print("🤖 Testing Individual Agents...")
    print("-" * 40)
    
    # 1. Test Prompt Parser Agent
    print("\n1️⃣ PROMPT PARSING AGENT")
    try:
        parser = PromptParsingAgent()
        parsed = await parser.parse_prompt(test_prompts[0])
        print(f"✅ Parsed prompt with intent: {parsed.intent}")
        print(f"   Entities: {len(parsed.entities)}, Modifiers: {len(parsed.modifiers)}, References: {len(parsed.references)}")
    except Exception as e:
        print(f"❌ Prompt parsing failed: {e}")
    
    # 2. Test Reference Retrieval Agent (if API keys available)
    print("\n2️⃣ REFERENCE RETRIEVAL AGENT")
    try:
        retriever = ReferenceRetrievalAgent()
        # Create a test reference
        from src.pic_arcade_agentic.types import PromptReference, PromptReferenceType
        test_ref = PromptReference(
            text="Emma Stone",
            type=PromptReferenceType.CELEBRITY,
            search_query="Emma Stone portrait photo",
            confidence=0.9
        )
        references = await retriever.retrieve_references([test_ref])
        print(f"✅ Retrieved references for {len(references)} items")
        if references and hasattr(references[0], 'image_urls'):
            print(f"   Found {len(references[0].image_urls)} images for first reference")
    except Exception as e:
        print(f"❌ Reference retrieval failed: {e}")
    
    # 3. Test Workflow Orchestrator
    print("\n3️⃣ WORKFLOW ORCHESTRATOR")
    try:
        orchestrator = WorkflowOrchestrator()
        result = await orchestrator.process_prompt(test_prompts[1])
        print(f"✅ Workflow completed with status: {result.status.value}")
        if result.context.prompt:
            print(f"   Intent: {result.context.prompt.intent}")
            print(f"   References found: {len(result.context.references) if result.context.references else 0}")
    except Exception as e:
        print(f"❌ Workflow orchestration failed: {e}")
    
    # 4. Test Tool-First Agent
    print("\n4️⃣ TOOL-FIRST AGENT")
    try:
        tool_agent = ToolFirstAgent()
        result = await tool_agent.process_request(test_prompts[2])
        print(f"✅ Tool-first processing: {'Success' if result['success'] else 'Failed'}")
        if result['success']:
            tools_used = result.get('metadata', {}).get('tools_used', [])
            print(f"   Tools used: {tools_used}")
    except Exception as e:
        print(f"❌ Tool-first processing failed: {e}")
    
    print("\n" + "=" * 60)
    print("📊 DECISION LOGGING ANALYSIS")
    print("=" * 60)
    
    # Display decision statistics
    stats = decision_logger.get_decision_stats()
    print(f"\n📈 Overall Statistics:")
    print(f"   Total Decisions: {stats['total_decisions']}")
    print(f"   Successful: {stats['successful_decisions']}")
    print(f"   Failed: {stats['failed_decisions']}")
    print(f"   Average Execution Time: {stats['average_execution_time']:.2f}ms")
    
    if stats['decision_types']:
        print(f"\n🔧 Decision Types:")
        for decision_type, count in stats['decision_types'].items():
            print(f"   {decision_type}: {count} decisions")
    
    # Display recent decisions
    history = decision_logger.get_decision_history()
    if history:
        print(f"\n📝 Recent Decisions ({len(history)} total):")
        for decision in history[-3:]:  # Show last 3 decisions
            print(f"\n   🔸 {decision.agent_name} (Request: {decision.request_id})")
            print(f"      Success: {'✅' if decision.success else '❌'}")
            print(f"      Steps: {decision.total_steps}")
            if decision.completed_at:
                duration = (decision.completed_at - decision.started_at) * 1000
                print(f"      Duration: {duration:.2f}ms")
            
            # Show decision steps
            if decision.steps:
                print(f"      Decision Steps:")
                for i, step in enumerate(decision.steps[:3]):  # Show first 3 steps
                    confidence = f" (confidence: {step.confidence_score:.2f})" if step.confidence_score else ""
                    print(f"        {i+1}. {step.decision_type.value}{confidence}")
                    print(f"           {step.decision_reasoning}")
                if len(decision.steps) > 3:
                    print(f"        ... and {len(decision.steps) - 3} more steps")
    
    # Export decision data
    print(f"\n💾 Exporting Decision Data...")
    export_path = decision_logger.export_decisions_to_json()
    print(f"   Exported to: {export_path}")
    
    # Show log file location
    if decision_logger.log_directory:
        log_dir = Path(decision_logger.log_directory)
        log_files = list(log_dir.glob("decisions_*.jsonl"))
        if log_files:
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            print(f"   Latest log file: {latest_log}")
    
    print(f"\n📋 How to Review Decision Logs:")
    print(f"   1. Check the exported JSON file for complete decision data")
    print(f"   2. View JSONL log files for real-time decision streaming")
    print(f"   3. Use decision_logger.get_decision_history() in your code")
    print(f"   4. Filter by agent: decision_logger.get_decision_history('AgentName')")
    
    print(f"\n🎯 Decision Types Available for Filtering:")
    for decision_type in DecisionType:
        print(f"   - {decision_type.value}")
    
    print(f"\n✨ Decision Logging Summary:")
    print(f"   Every agent decision is now tracked with:")
    print(f"   • Input data and reasoning")
    print(f"   • Confidence scores")
    print(f"   • Execution timing")
    print(f"   • Error handling")
    print(f"   • Complete audit trail")
    
    return stats


def review_decision_logs():
    """
    Function to review and analyze decision logs.
    """
    print("\n🔍 DECISION LOG REVIEW UTILITY")
    print("=" * 40)
    
    # Get all decisions
    history = decision_logger.get_decision_history()
    
    if not history:
        print("No decision history found. Run the demonstration first!")
        return
    
    # Group by agent
    agents = {}
    for decision in history:
        if decision.agent_name not in agents:
            agents[decision.agent_name] = []
        agents[decision.agent_name].append(decision)
    
    print(f"📊 Decisions by Agent:")
    for agent_name, decisions in agents.items():
        successful = sum(1 for d in decisions if d.success)
        failed = len(decisions) - successful
        avg_steps = sum(d.total_steps for d in decisions) / len(decisions)
        
        print(f"\n  🤖 {agent_name}:")
        print(f"     Total Decisions: {len(decisions)}")
        print(f"     Success Rate: {successful}/{len(decisions)} ({successful/len(decisions)*100:.1f}%)")
        print(f"     Average Steps: {avg_steps:.1f}")
    
    # Show decision timeline
    print(f"\n⏰ Decision Timeline (last 5 decisions):")
    for decision in history[-5:]:
        timestamp = datetime.fromtimestamp(decision.started_at).strftime("%H:%M:%S")
        status = "✅" if decision.success else "❌"
        print(f"  {timestamp} - {decision.agent_name} {status} ({decision.total_steps} steps)")


async def main():
    """Main demonstration function."""
    try:
        # Run the demonstration
        await demonstrate_decision_logging()
        
        # Review the logs
        review_decision_logs()
        
        print(f"\n🎉 Decision logging demonstration completed!")
        print(f"💡 All agent decisions are now comprehensively tracked for review.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 