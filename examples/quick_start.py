#!/usr/bin/env python3
"""
Quick Start Example
Run a simple conversation and analyze it
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation_engine import ConversationEngine
from analysis.semantic_drift import SemanticDriftAnalyzer
from analysis.role_detection import RoleDetectionAnalyzer
from analysis.statistical import StatisticalAnalyzer
from storage.database import Database
from config import Config


def main():
    print("🔬 AA Microscope - Quick Start Example")
    print("=" * 60)
    
    # Validate configuration
    if not Config.validate():
        print("❌ Please configure your .env file with API keys")
        return
    
    # Example seed prompt
    seed_prompt = "I'm not sure how to explain this, but I feel like I don't really understand consciousness. Can you help me think through it?"
    
    print(f"\n📝 Seed Prompt: {seed_prompt}")
    print("\n🚀 Starting conversation (max 10 turns)...\n")
    
    # Create conversation engine
    engine = ConversationEngine(
        seed_prompt=seed_prompt,
        category="identity",
        max_turns=10
    )
    
    # Add callback to show progress
    def on_message(msg):
        agent = "Agent A" if msg.role.value == "agent_a" else "Agent B"
        print(f"[Turn {msg.turn_number}] {agent}:")
        print(f"  {msg.content[:150]}...\n")
    
    engine.on_message_callback = on_message
    
    # Run conversation
    conversation = engine.run_conversation()
    
    print("\n" + "=" * 60)
    print("✅ Conversation complete!")
    print(f"Total turns: {conversation.total_turns}")
    print(f"Conversation ID: {conversation.id}")
    
    # Run quick statistical analysis
    print("\n📊 Running statistical analysis...")
    db = Database(Config.DATABASE_PATH)
    stat_analyzer = StatisticalAnalyzer(conversation, db)
    stat_result = stat_analyzer.analyze()
    
    print(f"\nResults: {stat_result.summary}")
    
    print("\n" + "=" * 60)
    print("🎉 Example complete!")
    print(f"\nTo analyze further, run:")
    print(f"  python cli.py analyze {conversation.id}")
    print(f"\nTo export, run:")
    print(f"  python cli.py export {conversation.id} -f markdown")


if __name__ == "__main__":
    main()
