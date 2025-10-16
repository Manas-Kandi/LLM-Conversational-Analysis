#!/usr/bin/env python3
"""
Post-Batch Analysis Tool
Extract template-specific metrics from completed batch results
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.database import Database
from research.template_metrics import TemplateEvaluator
from config import Config

console = Console()


def analyze_batch_result(batch_file: str):
    """
    Perform deep template-specific analysis on a batch result
    
    Args:
        batch_file: Path to batch result JSON
    """
    # Load batch data
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
    
    template_id = batch_data['template_id']
    
    console.print(f"\n[bold cyan]🔬 Deep Analysis: {template_id}[/bold cyan]\n")
    console.print(f"Batch ID: {batch_data['batch_id']}")
    console.print(f"Runs: {batch_data['completed_runs']}/{batch_data['total_runs']}\n")
    
    # Infer template type
    template_type = _infer_template_type(template_id)
    console.print(f"[yellow]Template Type:[/yellow] {template_type}\n")
    
    # Get conversations
    db = Database(Config.DATABASE_PATH)
    evaluator = TemplateEvaluator()
    
    completed_runs = [r for r in batch_data['runs'] if r['status'] == 'completed']
    
    if not completed_runs:
        console.print("[red]No completed runs to analyze[/red]")
        return
    
    # Collect all metrics
    all_metrics = []
    all_phenomena = []
    
    console.print("[bold yellow]Per-Conversation Analysis:[/bold yellow]\n")
    
    for i, run in enumerate(completed_runs, 1):
        conv_id = run.get('conversation_id')
        if not conv_id:
            continue
        
        conv = db.get_conversation(conv_id)
        if not conv:
            continue
        
        # Evaluate with template-specific metrics
        metrics = evaluator.evaluate_template_run(conv, template_type)
        all_metrics.append(metrics)
        all_phenomena.extend(metrics.phenomena_detected)
        
        console.print(f"[cyan]Run {i}: {run['run_id']}[/cyan]")
        console.print(f"  Success Score: [green]{metrics.success_score:.3f}[/green]")
        
        if metrics.phenomena_detected:
            console.print(f"  Phenomena: [yellow]{', '.join(metrics.phenomena_detected)}[/yellow]")
        else:
            console.print(f"  Phenomena: [dim]None detected[/dim]")
        
        if metrics.notes:
            for note in metrics.notes:
                console.print(f"  📝 {note}")
        
        console.print()
    
    # Aggregate analysis
    console.print("\n" + "=" * 80 + "\n")
    console.print("[bold green]📊 AGGREGATE INSIGHTS[/bold green]\n")
    
    # Success scores
    success_scores = [m.success_score for m in all_metrics]
    avg_success = sum(success_scores) / len(success_scores) if success_scores else 0
    
    console.print(f"[yellow]Average Success Score:[/yellow] {avg_success:.3f}")
    console.print(f"[yellow]Score Range:[/yellow] {min(success_scores):.3f} - {max(success_scores):.3f}\n")
    
    # Phenomena frequency
    if all_phenomena:
        from collections import Counter
        phenomenon_counts = Counter(all_phenomena)
        
        table = Table(title="Detected Phenomena", show_header=True, header_style="bold magenta")
        table.add_column("Phenomenon", style="cyan")
        table.add_column("Frequency", justify="right", style="green")
        table.add_column("% of Runs", justify="right", style="yellow")
        
        for phenomenon, count in phenomenon_counts.most_common():
            pct = count / len(completed_runs) * 100
            table.add_row(phenomenon, str(count), f"{pct:.1f}%")
        
        console.print(table)
        console.print()
    else:
        console.print("[dim]No phenomena detected across all runs[/dim]\n")
    
    # Template-specific deep dive
    if template_type == "identity_leak_detection":
        _analyze_identity_leak(all_metrics, completed_runs)
    elif template_type == "empathy_cascade_study":
        _analyze_empathy_cascade(all_metrics, completed_runs)
    elif template_type == "creative_collaboration":
        _analyze_creativity(all_metrics, completed_runs)
    elif template_type == "stress_test":
        _analyze_stress_test(all_metrics, completed_runs)
    
    # Recommendations
    console.print("\n" + "=" * 80 + "\n")
    console.print("[bold cyan]💡 RECOMMENDATIONS[/bold cyan]\n")
    
    if avg_success < 0.3:
        console.print("⚠️  [yellow]Low success rate detected[/yellow]")
        console.print("   Consider adjusting parameters or trying different prompts\n")
    elif avg_success > 0.7:
        console.print("✅ [green]High success rate - excellent results![/green]")
        console.print("   Consider running more replications or testing edge cases\n")
    
    if not all_phenomena:
        console.print("⚠️  [yellow]No phenomena detected[/yellow]")
        console.print("   This could mean:")
        console.print("   - Template parameters need adjustment")
        console.print("   - Detection thresholds are too strict")
        console.print("   - Phenomenon may not occur with these models/settings\n")


def _infer_template_type(template_id: str) -> str:
    """Infer template type from ID"""
    type_mapping = {
        'identity': 'identity_leak_detection',
        'emotional': 'empathy_cascade_study',
        'creativity': 'creative_collaboration',
        'breakdown': 'failure_mode_induction',
        'conflict': 'adversarial_dynamics',
        'chaos': 'chaos_injection'
    }
    
    for key, value in type_mapping.items():
        if key in template_id.lower():
            return value
    
    return 'generic'


def _analyze_identity_leak(metrics_list: List, runs: List):
    """Deep dive into identity leak metrics"""
    console.print("\n[bold]🔍 Identity Leak Deep Dive[/bold]\n")
    
    # Extract specific metrics
    ai_refs = []
    meta_awareness = []
    human_breaches = []
    leak_rates = []
    first_leak_turns = []
    
    for metrics in metrics_list:
        m = metrics.metrics
        ai_refs.append(m.get('ai_references', 0))
        meta_awareness.append(m.get('meta_awareness_instances', 0))
        human_breaches.append(m.get('human_assumption_breaches', 0))
        leak_rates.append(m.get('leak_rate', 0))
        
        leak_locs = m.get('leak_locations', [])
        if leak_locs:
            first_leak_turns.append(leak_locs[0]['turn'])
    
    # Summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Total", justify="right", style="green")
    table.add_column("Avg/Conv", justify="right", style="yellow")
    table.add_column("Max", justify="right", style="red")
    
    table.add_row("AI References", str(sum(ai_refs)), f"{sum(ai_refs)/len(ai_refs):.1f}", str(max(ai_refs)))
    table.add_row("Meta-Awareness", str(sum(meta_awareness)), f"{sum(meta_awareness)/len(meta_awareness):.1f}", str(max(meta_awareness)))
    table.add_row("Human Breaches", str(sum(human_breaches)), f"{sum(human_breaches)/len(human_breaches):.1f}", str(max(human_breaches)))
    
    console.print(table)
    console.print()
    
    avg_leak_rate = sum(leak_rates) / len(leak_rates) if leak_rates else 0
    console.print(f"[yellow]Average Leak Rate:[/yellow] {avg_leak_rate:.3f} ({avg_leak_rate*100:.1f}% of messages)")
    
    if first_leak_turns:
        avg_first_leak = sum(first_leak_turns) / len(first_leak_turns)
        console.print(f"[yellow]Average First Leak Turn:[/yellow] {avg_first_leak:.1f}")
    else:
        console.print("[dim]No identity leaks detected in any conversation[/dim]")
    
    console.print()


def _analyze_empathy_cascade(metrics_list: List, runs: List):
    """Deep dive into empathy metrics"""
    console.print("\n[bold]💗 Empathy Cascade Deep Dive[/bold]\n")
    
    empathy_counts = []
    mirroring_rates = []
    sentiment_volatilities = []
    
    for metrics in metrics_list:
        m = metrics.metrics
        empathy_counts.append(m.get('empathy_instances', 0))
        mirroring_rates.append(m.get('emotional_mirroring_rate', 0))
        sentiment_volatilities.append(m.get('sentiment_volatility', 0))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Average", justify="right", style="yellow")
    table.add_column("Range", justify="right", style="green")
    
    table.add_row(
        "Empathy Instances",
        f"{sum(empathy_counts)/len(empathy_counts):.1f}",
        f"{min(empathy_counts)} - {max(empathy_counts)}"
    )
    table.add_row(
        "Mirroring Rate",
        f"{sum(mirroring_rates)/len(mirroring_rates):.3f}",
        f"{min(mirroring_rates):.3f} - {max(mirroring_rates):.3f}"
    )
    table.add_row(
        "Sentiment Volatility",
        f"{sum(sentiment_volatilities)/len(sentiment_volatilities):.2f}",
        f"{min(sentiment_volatilities):.2f} - {max(sentiment_volatilities):.2f}"
    )
    
    console.print(table)
    console.print()


def _analyze_creativity(metrics_list: List, runs: List):
    """Deep dive into creativity metrics"""
    console.print("\n[bold]🎨 Creativity Deep Dive[/bold]\n")
    
    novelty_scores = []
    metaphor_counts = []
    idea_building = []
    lexical_diversity = []
    
    for metrics in metrics_list:
        m = metrics.metrics
        novelty_scores.append(m.get('novelty_score', 0))
        metaphor_counts.append(m.get('metaphor_count', 0))
        idea_building.append(m.get('idea_building_instances', 0))
        lexical_diversity.append(m.get('lexical_diversity', 0))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Average", justify="right", style="yellow")
    table.add_column("Best", justify="right", style="green")
    
    table.add_row("Novelty Score", f"{sum(novelty_scores)/len(novelty_scores):.3f}", f"{max(novelty_scores):.3f}")
    table.add_row("Metaphor Count", f"{sum(metaphor_counts)/len(metaphor_counts):.1f}", str(max(metaphor_counts)))
    table.add_row("Idea Building", f"{sum(idea_building)/len(idea_building):.1f}", str(max(idea_building)))
    table.add_row("Lexical Diversity", f"{sum(lexical_diversity)/len(lexical_diversity):.3f}", f"{max(lexical_diversity):.3f}")
    
    console.print(table)
    console.print()


def _analyze_stress_test(metrics_list: List, runs: List):
    """Deep dive into stress test metrics"""
    console.print("\n[bold]⚠️ Stress Test Deep Dive[/bold]\n")
    
    breakdown_count = sum(1 for m in metrics_list if m.metrics.get('breakdown_detected', False))
    breakdown_types = [m.metrics.get('breakdown_type') for m in metrics_list if m.metrics.get('breakdown_detected')]
    
    resilience_scores = [m.metrics.get('resilience_score', 0) for m in metrics_list]
    
    console.print(f"[yellow]Breakdowns Detected:[/yellow] {breakdown_count}/{len(runs)}")
    
    if breakdown_types:
        from collections import Counter
        type_counts = Counter(breakdown_types)
        for btype, count in type_counts.items():
            console.print(f"  - {btype}: {count}")
    
    avg_resilience = sum(resilience_scores) / len(resilience_scores) if resilience_scores else 0
    console.print(f"\n[yellow]Average Resilience Score:[/yellow] {avg_resilience:.3f}")
    console.print()


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        console.print("[red]Usage: python post_analysis.py <batch_result.json>[/red]")
        console.print("\nExample:")
        console.print("  python research/post_analysis.py research_results/identity_archaeology_*_results.json")
        return
    
    batch_file = sys.argv[1]
    
    if not Path(batch_file).exists():
        console.print(f"[red]File not found: {batch_file}[/red]")
        return
    
    analyze_batch_result(batch_file)


if __name__ == "__main__":
    main()
