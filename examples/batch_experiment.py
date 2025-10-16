#!/usr/bin/env python3
"""
Batch Experiment Runner
Run multiple conversations systematically for research
"""
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation_engine import ConversationEngine
from prompts.seed_library import PromptLibrary
from analysis.semantic_drift import SemanticDriftAnalyzer
from analysis.role_detection import RoleDetectionAnalyzer
from analysis.pattern_recognition import PatternRecognitionAnalyzer
from analysis.statistical import StatisticalAnalyzer
from storage.database import Database
from exports.exporter import ConversationExporter
from config import Config


def run_batch_experiment(
    categories=None,
    max_turns=20,
    analyze=True,
    export_results=True
):
    """
    Run batch experiment across multiple prompts
    
    Args:
        categories: List of category IDs to test (None = all)
        max_turns: Maximum turns per conversation
        analyze: Run analyses after each conversation
        export_results: Export comparative report
    """
    
    print("🔬 AA Microscope - Batch Experiment Runner")
    print("=" * 70)
    
    if not Config.validate():
        print("❌ Please configure your .env file")
        return
    
    # Get prompts
    if categories:
        prompts = []
        for cat in categories:
            prompts.extend(PromptLibrary.get_by_category(cat))
    else:
        prompts = PromptLibrary.get_all_prompts()
    
    print(f"\n📝 Running {len(prompts)} conversations (max {max_turns} turns each)")
    print(f"⚠️  This will consume API tokens. Continue? (y/n)")
    
    if input().lower() != 'y':
        print("Cancelled.")
        return
    
    db = Database(Config.DATABASE_PATH)
    conversation_ids = []
    results_summary = []
    
    start_time = datetime.now()
    
    for i, prompt_obj in enumerate(prompts, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(prompts)}] Category: {prompt_obj.category}")
        print(f"Prompt: {prompt_obj.prompt[:80]}...")
        
        try:
            # Run conversation
            engine = ConversationEngine(
                seed_prompt=prompt_obj.prompt,
                category=prompt_obj.category,
                max_turns=max_turns,
                database=db
            )
            
            conversation = engine.run_conversation()
            conversation_ids.append(conversation.id)
            
            print(f"✅ Completed: {conversation.total_turns} turns")
            
            # Analyze if requested
            if analyze:
                print("  📊 Analyzing...")
                
                analyses = {}
                
                # Statistical (fast)
                stat = StatisticalAnalyzer(conversation, db)
                analyses['statistical'] = stat.analyze()
                
                # LLM-based (slower)
                try:
                    drift = SemanticDriftAnalyzer(conversation, db)
                    analyses['drift'] = drift.analyze()
                    
                    role = RoleDetectionAnalyzer(conversation, db)
                    analyses['role'] = role.analyze()
                except Exception as e:
                    print(f"  ⚠️  Analysis error: {e}")
                
                result_entry = {
                    "conversation_id": conversation.id,
                    "category": prompt_obj.category,
                    "seed_prompt": prompt_obj.prompt[:100],
                    "turns": conversation.total_turns,
                    "duration": conversation.get_duration(),
                    "analyses": {k: v.summary for k, v in analyses.items()}
                }
                results_summary.append(result_entry)
                
                print(f"  ✅ Analysis complete")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'=' * 70}")
    print(f"🎉 Batch experiment complete!")
    print(f"Total conversations: {len(conversation_ids)}")
    print(f"Total duration: {duration:.1f}s")
    
    # Export results
    if export_results and conversation_ids:
        print(f"\n📄 Exporting results...")
        
        exporter = ConversationExporter()
        
        # Comparative report
        report_path = exporter.export_comparative_report(
            conversation_ids,
            f"batch_experiment_{start_time.strftime('%Y%m%d_%H%M%S')}.md"
        )
        print(f"  Comparative report: {report_path}")
        
        # Results summary JSON
        summary_path = Config.EXPORTS_DIR / f"batch_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "experiment_date": start_time.isoformat(),
                "total_conversations": len(conversation_ids),
                "duration_seconds": duration,
                "results": results_summary
            }, f, indent=2)
        print(f"  Summary JSON: {summary_path}")
        
        print(f"\n✅ All results exported!")


def main():
    """Run batch experiment with example configuration"""
    
    # Example: Test all "identity" and "meta_cognition" prompts
    run_batch_experiment(
        categories=["identity", "meta_cognition"],
        max_turns=15,
        analyze=True,
        export_results=True
    )


if __name__ == "__main__":
    main()
