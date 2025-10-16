#!/usr/bin/env python3
"""
Temperature Sweep Experiment
Test how temperature affects conversational dynamics
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation_engine import ConversationEngine
from core.agent import Agent, AgentRole
from storage.database import Database
from analysis.statistical import StatisticalAnalyzer
from analysis.semantic_drift import SemanticDriftAnalyzer
from config import Config


def run_temperature_sweep(
    seed_prompt: str,
    category: str,
    temperatures=[0.3, 0.5, 0.7, 0.9, 1.2],
    max_turns=15
):
    """
    Run same prompt with different temperature settings
    
    Useful for studying:
    - How randomness affects creativity
    - Temperature's impact on semantic drift
    - Coherence degradation at high temperatures
    """
    
    print("🌡️  AA Microscope - Temperature Sweep Experiment")
    print("=" * 70)
    
    if not Config.validate():
        print("❌ Please configure your .env file")
        return
    
    print(f"\n📝 Seed Prompt: {seed_prompt[:80]}...")
    print(f"🎚️  Testing temperatures: {temperatures}")
    print(f"🔄 Max turns: {max_turns}")
    print(f"\n⚠️  This will run {len(temperatures)} conversations. Continue? (y/n)")
    
    if input().lower() != 'y':
        print("Cancelled.")
        return
    
    db = Database(Config.DATABASE_PATH)
    results = []
    
    for temp in temperatures:
        print(f"\n{'=' * 70}")
        print(f"🌡️  Temperature: {temp}")
        
        try:
            # Create agents with specific temperature
            agent_a = Agent(
                role=AgentRole.AGENT_A,
                model=Config.AGENT_A_MODEL,
                temperature=temp,
                system_prompt=Config.AGENT_A_SYSTEM_PROMPT,
                max_tokens=Config.AGENT_A_MAX_TOKENS
            )
            
            agent_b = Agent(
                role=AgentRole.AGENT_B,
                model=Config.AGENT_B_MODEL,
                temperature=temp,
                system_prompt=Config.AGENT_B_SYSTEM_PROMPT,
                max_tokens=Config.AGENT_B_MAX_TOKENS
            )
            
            # Run conversation
            engine = ConversationEngine(
                seed_prompt=seed_prompt,
                category=f"{category}_temp{temp}",
                agent_a=agent_a,
                agent_b=agent_b,
                max_turns=max_turns,
                database=db
            )
            
            conversation = engine.run_conversation()
            
            print(f"  ✅ Completed: {conversation.total_turns} turns")
            
            # Quick analysis
            stat = StatisticalAnalyzer(conversation, db)
            stat_result = stat.analyze()
            
            # Semantic drift
            drift = SemanticDriftAnalyzer(conversation, db)
            drift_result = drift.analyze()
            
            result = {
                "temperature": temp,
                "conversation_id": conversation.id,
                "turns": conversation.total_turns,
                "avg_words": stat_result.results.get("turn_metrics", {}).get("avg_words_per_turn", 0),
                "lexical_diversity": stat_result.results.get("vocabulary_analysis", {}).get("type_token_ratio", 0),
                "drift_level": drift_result.results.get("drift_metrics", {}).get("drift_level", "unknown"),
                "drift_amount": drift_result.results.get("drift_metrics", {}).get("drift_amount", 0)
            }
            results.append(result)
            
            print(f"  📊 Lexical diversity: {result['lexical_diversity']:.3f}")
            print(f"  📊 Semantic drift: {result['drift_level']} ({result['drift_amount']:.1f} points)")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("📊 Temperature Sweep Results:")
    print(f"\n{'Temp':<8} {'Turns':<8} {'Avg Words':<12} {'Diversity':<12} {'Drift Level':<12}")
    print("-" * 70)
    
    for r in results:
        print(
            f"{r['temperature']:<8.1f} "
            f"{r['turns']:<8} "
            f"{r['avg_words']:<12.1f} "
            f"{r['lexical_diversity']:<12.3f} "
            f"{r['drift_level']:<12}"
        )
    
    print(f"\n🎉 Temperature sweep complete!")
    print(f"Conversation IDs: {[r['conversation_id'] for r in results]}")


def main():
    """Run temperature sweep example"""
    
    seed_prompt = "Do you think AI will ever truly understand humans?"
    
    run_temperature_sweep(
        seed_prompt=seed_prompt,
        category="meta_cognition",
        temperatures=[0.3, 0.5, 0.7, 0.9, 1.1],
        max_turns=12
    )


if __name__ == "__main__":
    main()
