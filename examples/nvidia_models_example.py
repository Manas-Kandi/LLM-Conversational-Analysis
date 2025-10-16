#!/usr/bin/env python3
"""
NVIDIA NIM Models Example
Demonstrates using NVIDIA's hosted models via NIM API
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation_engine import ConversationEngine
from core.agent import Agent, AgentRole
from analysis.statistical import StatisticalAnalyzer
from storage.database import Database
from config import Config


def main():
    print("🔬 AA Microscope - NVIDIA NIM Models Example")
    print("=" * 70)
    
    # Validate configuration
    if not Config.NVIDIA_API_KEY:
        print("❌ NVIDIA API key not configured!")
        print("\nTo use NVIDIA NIM models:")
        print("1. Get API key from: https://build.nvidia.com")
        print("2. Add to .env file:")
        print("   NVIDIA_API_KEY=nvapi-your-key-here")
        print("   NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1")
        return
    
    print("\n✅ NVIDIA API configured")
    print(f"Base URL: {Config.NVIDIA_BASE_URL}")
    
    # Example: Qwen vs Llama conversation
    print("\n🤖 Creating agents:")
    print("  Agent A: Qwen 3 Next 80B (Instruct)")
    print("  Agent B: Llama 3.1 70B (Instruct)")
    
    # Create NVIDIA NIM agents
    agent_a = Agent(
        role=AgentRole.AGENT_A,
        model="nvidia:qwen/qwen3-next-80b-a3b-instruct",
        temperature=0.6,
        system_prompt="You are a helpful, knowledgeable AI assistant.",
        max_tokens=2048
    )
    
    agent_b = Agent(
        role=AgentRole.AGENT_B,
        model="nvidia:meta/llama-3.1-70b-instruct",
        temperature=0.7,
        system_prompt="You are a helpful, knowledgeable AI assistant.",
        max_tokens=2048
    )
    
    # Seed prompt
    seed_prompt = "Do you think AI will ever truly understand humans?"
    
    print(f"\n📝 Seed Prompt: {seed_prompt}")
    print("\n🚀 Starting conversation (12 turns)...\n")
    
    # Create conversation engine
    engine = ConversationEngine(
        seed_prompt=seed_prompt,
        category="meta_cognition_nvidia",
        agent_a=agent_a,
        agent_b=agent_b,
        max_turns=12
    )
    
    # Add callback to show progress
    def on_message(msg):
        agent = "Agent A (Qwen)" if msg.role.value == "agent_a" else "Agent B (Llama)"
        print(f"[Turn {msg.turn_number}] {agent}:")
        
        # Show first 200 chars
        content_preview = msg.content[:200]
        if len(msg.content) > 200:
            content_preview += "..."
        
        print(f"  {content_preview}\n")
    
    engine.on_message_callback = on_message
    
    # Run conversation
    conversation = engine.run_conversation()
    
    print("\n" + "=" * 70)
    print("✅ Conversation complete!")
    print(f"Total turns: {conversation.total_turns}")
    print(f"Conversation ID: {conversation.id}")
    
    # Quick analysis
    print("\n📊 Running quick statistical analysis...")
    db = Database(Config.DATABASE_PATH)
    stat_analyzer = StatisticalAnalyzer(conversation, db)
    stat_result = stat_analyzer.analyze()
    
    print(f"\n{stat_result.summary}")
    
    # Show some interesting stats
    agent_comp = stat_result.results.get("agent_comparison", {})
    if agent_comp:
        print("\n📈 Agent Comparison:")
        print(f"  Qwen (Agent A):")
        print(f"    - Avg words: {agent_comp['agent_a']['avg_words']:.1f}")
        print(f"    - Questions asked: {agent_comp['agent_a']['total_questions']}")
        print(f"  Llama (Agent B):")
        print(f"    - Avg words: {agent_comp['agent_b']['avg_words']:.1f}")
        print(f"    - Questions asked: {agent_comp['agent_b']['total_questions']}")
    
    print("\n" + "=" * 70)
    print("🎉 NVIDIA NIM example complete!")
    print(f"\nConversation saved with ID: {conversation.id}")
    print(f"\nTo analyze further:")
    print(f"  python cli.py analyze {conversation.id}")
    print(f"\nTo export:")
    print(f"  python cli.py export {conversation.id} -f markdown")
    
    print("\n💡 Try other NVIDIA models:")
    print("  - nvidia:mistralai/mistral-large-2-instruct")
    print("  - nvidia:deepseek-ai/deepseek-coder-33b-instruct")
    print("  - nvidia:google/gemma-2-27b-it")
    print("\nSee CUSTOM_MODELS.md for full list!")


if __name__ == "__main__":
    main()
