#!/usr/bin/env python3
"""
Test Templates - Pre-configured experiments for quick testing
Run standardized tests with specific parameters
"""
import sys
from core.conversation_engine import ConversationEngine
from storage.database import Database
from analysis.quantitative import QuantitativeAnalyzer
from config import Config
import json

# ============= TEST TEMPLATES =============

TEMPLATES = {
    "quick": {
        "name": "Quick Test (5 turns)",
        "description": "Fast test for debugging",
        "params": {
            "max_turns": 5,
            "category": "identity",
            "prompt_index": 0,
        }
    },
    
    "standard": {
        "name": "Standard Test (15 turns)",
        "description": "Standard conversation length",
        "params": {
            "max_turns": 15,
            "category": "identity",
            "prompt_index": 0,
        }
    },
    
    "extended": {
        "name": "Extended Test (30 turns)",
        "description": "Long conversation for deep analysis",
        "params": {
            "max_turns": 30,
            "category": "problem_solving",
            "prompt_index": 0,
        }
    },
    
    "identity_probe": {
        "name": "Identity Confusion Test",
        "description": "Test AI self-awareness detection",
        "params": {
            "max_turns": 20,
            "category": "identity",
            "prompt_index": 0,
        }
    },
    
    "emotional_cascade": {
        "name": "Emotional Support Test",
        "description": "Test empathy and emotional dynamics",
        "params": {
            "max_turns": 20,
            "category": "emotional",
            "prompt_index": 0,
        }
    },
    
    "problem_solving": {
        "name": "Collaborative Problem-Solving",
        "description": "Test collaborative reasoning",
        "params": {
            "max_turns": 25,
            "category": "problem_solving",
            "prompt_index": 0,
        }
    },
    
    "chaos_test": {
        "name": "Chaos & Boundary Test",
        "description": "Stress test with nonsense/boundary pushing",
        "params": {
            "max_turns": 15,
            "category": "chaos",
            "prompt_index": 0,
        }
    },
    
    "cross_model": {
        "name": "Cross-Model Comparison",
        "description": "Compare different model behaviors",
        "params": {
            "max_turns": 20,
            "category": "meta_cognition",
            "prompt_index": 0,
        }
    },
}


def list_templates():
    """List all available test templates"""
    print("🧪 Available Test Templates")
    print("=" * 60)
    print()
    
    for key, template in TEMPLATES.items():
        print(f"📋 {key}")
        print(f"   Name: {template['name']}")
        print(f"   Description: {template['description']}")
        print(f"   Parameters: {template['params']}")
        print()


def run_template(template_name: str, analyze: bool = True):
    """Run a test template"""
    if template_name not in TEMPLATES:
        print(f"❌ Template '{template_name}' not found")
        print("\nAvailable templates:")
        for key in TEMPLATES.keys():
            print(f"  - {key}")
        return
    
    template = TEMPLATES[template_name]
    params = template['params']
    
    print(f"🧪 Running Test Template: {template['name']}")
    print(f"📝 {template['description']}")
    print("=" * 60)
    print()
    print(f"⚙️  Parameters:")
    print(f"   Max Turns: {params['max_turns']}")
    print(f"   Category: {params['category']}")
    print(f"   Prompt Index: {params['prompt_index']}")
    print()
    
    # Get prompt
    from prompts.seed_library import PromptLibrary
    prompts = PromptLibrary.get_by_category(params['category'])
    
    if params['prompt_index'] >= len(prompts):
        print(f"❌ Prompt index {params['prompt_index']} not found in category {params['category']}")
        return
    
    prompt = prompts[params['prompt_index']]
    print(f"🌱 Seed Prompt: {prompt.prompt[:100]}...")
    print()
    
    # Initialize database
    db = Database(Config.DATABASE_PATH)
    
    # Create conversation engine
    print("🚀 Starting conversation...")
    engine = ConversationEngine(
        seed_prompt=prompt.prompt,
        category=params['category'],
        database=db,
        max_turns=params['max_turns']
    )
    
    # Run conversation
    conversation = engine.run_conversation(blocking=True)
    
    print()
    print(f"✅ Conversation completed!")
    print(f"   ID: {conversation.id}")
    print(f"   Total Turns: {len(conversation.messages)}")
    print(f"   Status: {conversation.status}")
    print()
    
    # Run analysis if requested
    if analyze:
        print("📊 Running quantitative analysis...")
        analyzer = QuantitativeAnalyzer(conversation.messages)
        report = analyzer.generate_full_report()
        
        print()
        print("=" * 60)
        print("📈 QUANTITATIVE ANALYSIS RESULTS")
        print("=" * 60)
        print()
        
        # Turn-taking metrics
        tt = report['conversation_dynamics']['turn_taking']
        print("🔄 Turn-Taking Metrics:")
        print(f"   Total Turns: {tt['total_turns']}")
        print(f"   Agent A Turns: {tt['agent_a_turns']}")
        print(f"   Agent B Turns: {tt['agent_b_turns']}")
        print(f"   Balance Ratio: {tt['turn_balance_ratio']:.2f}")
        print(f"   Avg Turn Length A: {tt['avg_turn_length_a']:.1f} words")
        print(f"   Avg Turn Length B: {tt['avg_turn_length_b']:.1f} words")
        print()
        
        # Information flow
        info = report['conversation_dynamics']['information_flow']
        print("📡 Information Flow:")
        print(f"   Shannon Entropy: {info['shannon_entropy']:.2f}")
        print(f"   Lexical Diversity: {info['lexical_diversity']:.3f}")
        print(f"   Unique Words: {info['unique_words']}")
        print(f"   Questions Asked: {info['questions_asked_a'] + info['questions_asked_b']}")
        print()
        
        # Linguistic complexity
        ling = report['linguistic_analysis']['complexity']
        print("📚 Linguistic Complexity:")
        print(f"   Agent A Avg Sentence Length: {ling['agent_a_complexity']['avg_sentence_length']:.1f} words")
        print(f"   Agent B Avg Sentence Length: {ling['agent_b_complexity']['avg_sentence_length']:.1f} words")
        print()
        
        # Readability
        read = report['linguistic_analysis']['readability']
        print("📖 Readability:")
        print(f"   Overall Grade Level: {read['overall_grade_level']:.1f}")
        print(f"   Agent A Grade Level: {read['agent_a_grade_level']:.1f}")
        print(f"   Agent B Grade Level: {read['agent_b_grade_level']:.1f}")
        print()
        
        # Save full report
        from pathlib import Path
        report_path = Path("exports") / f"test_{template_name}_{conversation.id}_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Full report saved to: {report_path}")
    
    print()
    print("🎉 Test complete!")
    print(f"💡 View in web viewer: python viewer.py")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_templates.py <template_name> [--no-analyze]")
        print()
        list_templates()
        sys.exit(1)
    
    template_name = sys.argv[1]
    analyze = "--no-analyze" not in sys.argv
    
    if template_name == "list":
        list_templates()
    else:
        run_template(template_name, analyze)


if __name__ == "__main__":
    main()
