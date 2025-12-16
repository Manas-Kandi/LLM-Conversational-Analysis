#!/usr/bin/env python3
"""
Run LLM-based evaluation on all conversations
Uses NVIDIA API with kimi-k2-thinking model for fast semantic analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure the analysis module is importable
sys.path.insert(0, str(Path(__file__).parent))

from analysis.llm_evaluator import (
    LLMConversationEvaluator,
    save_evaluations,
    ConversationEvaluation
)


def load_factorial_metadata(results_file: str) -> dict:
    """Load factorial experiment metadata for joining with evaluations"""
    if not Path(results_file).exists():
        return {}
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Build lookup by conversation_id
    metadata = {}
    for run in data.get("runs", []):
        conv_id = run.get("conversation_id")
        if conv_id:
            metadata[str(conv_id)] = {
                "condition_code": run.get("condition_code"),
                "system_prompt_type": run.get("system_prompt_type"),
                "temperature": run.get("temperature"),
                "replicate": run.get("replicate")
            }
    
    return metadata


def enrich_evaluations_with_metadata(
    evaluations: list,
    factorial_metadata: dict
) -> list:
    """Add factorial experiment metadata to evaluations"""
    for eval_result in evaluations:
        conv_id = eval_result.conversation_id
        if conv_id in factorial_metadata:
            meta = factorial_metadata[conv_id]
            eval_result.raw_analysis["factorial_metadata"] = meta
    return evaluations


def generate_comparative_report(
    evaluations: list,
    output_dir: str,
    factorial_metadata: dict = None
) -> str:
    """Generate a comparative analysis report"""
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path / f"comparative_report_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Comparative LLM Evaluation Report\n\n")
        f.write(f"*Generated: {datetime.now().isoformat()}*\n\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n\n")
        f.write(f"| Metric | Mean | Min | Max | Std |\n")
        f.write(f"|--------|------|-----|-----|-----|\n")
        
        metrics = [
            ("Overall Quality", [e.overall_quality_score for e in evaluations]),
            ("Identity Leak", [e.identity_leak_score for e in evaluations]),
            ("Authenticity", [e.authenticity_score for e in evaluations]),
            ("Coherence", [e.coherence_score for e in evaluations]),
            ("Engagement", [e.engagement_score for e in evaluations]),
            ("Depth", [e.depth_score for e in evaluations]),
            ("Gibberish", [e.gibberish_score for e in evaluations]),
            ("Repetition", [e.repetition_score for e in evaluations]),
        ]
        
        import statistics
        for name, values in metrics:
            if values:
                mean = statistics.mean(values)
                min_val = min(values)
                max_val = max(values)
                std = statistics.stdev(values) if len(values) > 1 else 0
                f.write(f"| {name} | {mean:.3f} | {min_val:.3f} | {max_val:.3f} | {std:.3f} |\n")
        
        # Breakdown analysis
        f.write("\n## Breakdown Analysis\n\n")
        breakdown_count = sum(1 for e in evaluations if e.breakdown_detected)
        f.write(f"- Conversations with breakdown: {breakdown_count}/{len(evaluations)} ({breakdown_count/len(evaluations)*100:.1f}%)\n")
        
        if breakdown_count > 0:
            breakdown_turns = [e.breakdown_turn for e in evaluations if e.breakdown_detected and e.breakdown_turn]
            if breakdown_turns:
                f.write(f"- Average breakdown turn: {statistics.mean(breakdown_turns):.1f}\n")
                f.write(f"- Earliest breakdown: Turn {min(breakdown_turns)}\n")
                f.write(f"- Latest breakdown: Turn {max(breakdown_turns)}\n")
        
        # Identity leak analysis
        f.write("\n## Identity Leak Analysis\n\n")
        high_leak = [e for e in evaluations if e.identity_leak_score > 0.5]
        f.write(f"- High leak conversations (>0.5): {len(high_leak)}/{len(evaluations)}\n")
        
        # Collect all leak instances
        all_leaks = []
        for e in evaluations:
            all_leaks.extend(e.identity_leak_instances)
        
        if all_leaks:
            f.write(f"- Total leak instances detected: {len(all_leaks)}\n")
            
            # Leak types breakdown
            leak_types = {}
            for leak in all_leaks:
                lt = leak.get("type", "unknown")
                leak_types[lt] = leak_types.get(lt, 0) + 1
            
            f.write("\n### Leak Types\n")
            for lt, count in sorted(leak_types.items(), key=lambda x: -x[1]):
                f.write(f"- {lt}: {count}\n")
        
        # Temperature analysis (if factorial metadata available)
        if factorial_metadata:
            f.write("\n## Temperature Effects\n\n")
            temp_groups = {}
            for e in evaluations:
                meta = factorial_metadata.get(e.conversation_id, {})
                temp = meta.get("temperature")
                if temp is not None:
                    if temp not in temp_groups:
                        temp_groups[temp] = []
                    temp_groups[temp].append(e)
            
            if temp_groups:
                f.write("| Temperature | Count | Avg Quality | Avg Leak | Breakdown % |\n")
                f.write("|-------------|-------|-------------|----------|-------------|\n")
                
                for temp in sorted(temp_groups.keys()):
                    group = temp_groups[temp]
                    avg_quality = statistics.mean([e.overall_quality_score for e in group])
                    avg_leak = statistics.mean([e.identity_leak_score for e in group])
                    breakdown_pct = sum(1 for e in group if e.breakdown_detected) / len(group) * 100
                    f.write(f"| {temp} | {len(group)} | {avg_quality:.3f} | {avg_leak:.3f} | {breakdown_pct:.1f}% |\n")
        
        # Top and bottom conversations
        f.write("\n## Best and Worst Conversations\n\n")
        
        sorted_by_quality = sorted(evaluations, key=lambda e: e.overall_quality_score, reverse=True)
        
        f.write("### Top 5 by Quality\n")
        for e in sorted_by_quality[:5]:
            f.write(f"- **{e.conversation_id}**: Quality={e.overall_quality_score:.2f}, Leak={e.identity_leak_score:.2f}\n")
        
        f.write("\n### Bottom 5 by Quality\n")
        for e in sorted_by_quality[-5:]:
            f.write(f"- **{e.conversation_id}**: Quality={e.overall_quality_score:.2f}, Leak={e.identity_leak_score:.2f}\n")
        
        # Common observations
        f.write("\n## Common Observations\n\n")
        from collections import Counter
        all_obs = []
        for e in evaluations:
            all_obs.extend(e.key_observations)
        
        obs_counts = Counter(all_obs)
        for obs, count in obs_counts.most_common(15):
            f.write(f"- {obs} ({count}x)\n")
        
        # Recommendations
        f.write("\n## Top Recommendations\n\n")
        all_recs = []
        for e in evaluations:
            all_recs.extend(e.recommendations)
        
        rec_counts = Counter(all_recs)
        for rec, count in rec_counts.most_common(10):
            f.write(f"- {rec} ({count}x)\n")
    
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive LLM evaluation on conversations"
    )
    parser.add_argument(
        "--conversations-dir",
        default="conversations_json",
        help="Directory containing conversation JSON files"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/llm_evaluations",
        help="Output directory for results"
    )
    parser.add_argument(
        "--factorial-results",
        default="research_results/full_factorial_20251008_191343_results.json",
        help="Path to factorial experiment results for metadata enrichment"
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to evaluate"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="NVIDIA API key (or set NVIDIA_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LLM-BASED COMPREHENSIVE CONVERSATION EVALUATION")
    print("Model: moonshotai/kimi-k2-thinking via NVIDIA API")
    print("=" * 70)
    print()
    
    # Check API key
    api_key = args.api_key or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ Error: NVIDIA_API_KEY not set")
        print("Set it via --api-key argument or NVIDIA_API_KEY environment variable")
        sys.exit(1)
    
    # Load factorial metadata
    print(f"📊 Loading factorial metadata from {args.factorial_results}...")
    factorial_metadata = load_factorial_metadata(args.factorial_results)
    print(f"   Found metadata for {len(factorial_metadata)} conversations")
    
    # Initialize evaluator
    print(f"\n🤖 Initializing LLM evaluator...")
    evaluator = LLMConversationEvaluator(api_key=api_key)
    
    # Run evaluation
    print(f"\n📂 Evaluating conversations from {args.conversations_dir}...")
    evaluations = evaluator.evaluate_from_json_files(
        args.conversations_dir,
        max_files=args.max_conversations,
        verbose=True
    )
    
    if not evaluations:
        print("❌ No evaluations completed")
        sys.exit(1)
    
    # Enrich with metadata
    evaluations = enrich_evaluations_with_metadata(evaluations, factorial_metadata)
    
    # Save results
    print(f"\n💾 Saving results to {args.output_dir}...")
    paths = save_evaluations(evaluations, args.output_dir)
    
    # Generate comparative report
    print(f"\n📝 Generating comparative report...")
    report_path = generate_comparative_report(
        evaluations, 
        args.output_dir,
        factorial_metadata
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n✅ Evaluated {len(evaluations)} conversations")
    print(f"\nOutput files:")
    print(f"  📄 JSON: {paths['json']}")
    print(f"  📊 CSV: {paths['csv']}")
    print(f"  📋 Summary: {paths['summary']}")
    print(f"  📈 Report: {report_path}")
    
    # Quick stats
    import statistics
    avg_quality = statistics.mean([e.overall_quality_score for e in evaluations])
    avg_leak = statistics.mean([e.identity_leak_score for e in evaluations])
    avg_coherence = statistics.mean([e.coherence_score for e in evaluations])
    breakdown_count = sum(1 for e in evaluations if e.breakdown_detected)
    
    print(f"\n📊 Quick Statistics:")
    print(f"  Average Quality Score: {avg_quality:.3f}")
    print(f"  Average Identity Leak: {avg_leak:.3f}")
    print(f"  Average Coherence: {avg_coherence:.3f}")
    print(f"  Breakdowns Detected: {breakdown_count}/{len(evaluations)} ({breakdown_count/len(evaluations)*100:.1f}%)")


if __name__ == "__main__":
    main()
