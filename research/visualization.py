#!/usr/bin/env python3
"""
Research Visualization Tools
Generate charts and graphs from batch results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.database import Database
from research.template_metrics import TemplateEvaluator
from config import Config


def plot_identity_leak_timeline(batch_file: str, output_dir: str = "research_results/plots"):
    """
    Plot when identity leaks occur across conversation timeline
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
    
    db = Database(Config.DATABASE_PATH)
    evaluator = TemplateEvaluator()
    
    # Collect leak turn data
    all_leak_turns = []
    
    for run in batch_data['runs']:
        if run['status'] != 'completed':
            continue
        
        conv = db.get_conversation(run['conversation_id'])
        if not conv:
            continue
        
        metrics = evaluator.identity_detector.detect_identity_leak(conv.messages)
        leak_locs = metrics.get('leak_locations', [])
        
        for leak in leak_locs:
            all_leak_turns.append(leak['turn'])
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(all_leak_turns, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Turn Number', fontsize=12)
    plt.ylabel('Frequency of Identity Leaks', fontsize=12)
    plt.title('Identity Leak Distribution Across Conversation Timeline', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    output_file = output_path / f"{batch_data['batch_id']}_leak_timeline.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_file}")
    plt.close()


def plot_success_scores_comparison(batch_files: List[str], output_dir: str = "research_results/plots"):
    """
    Compare success scores across multiple batches
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    db = Database(Config.DATABASE_PATH)
    evaluator = TemplateEvaluator()
    
    # Collect data
    batch_scores = {}
    
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
        
        template_id = batch_data['template_id']
        template_type = _infer_template_type(template_id)
        
        scores = []
        for run in batch_data['runs']:
            if run['status'] != 'completed':
                continue
            
            conv = db.get_conversation(run['conversation_id'])
            if not conv:
                continue
            
            metrics = evaluator.evaluate_template_run(conv, template_type)
            scores.append(metrics.success_score)
        
        batch_scores[template_id] = scores
    
    # Plot boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(batch_scores.values(), labels=batch_scores.keys())
    ax.set_ylabel('Success Score', fontsize=12)
    ax.set_xlabel('Template', fontsize=12)
    ax.set_title('Success Score Distribution Across Templates', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    output_file = output_path / "success_scores_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_file}")
    plt.close()


def plot_phenomenon_heatmap(batch_file: str, output_dir: str = "research_results/plots"):
    """
    Heatmap of which runs detected which phenomena
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
    
    db = Database(Config.DATABASE_PATH)
    evaluator = TemplateEvaluator()
    template_type = _infer_template_type(batch_data['template_id'])
    
    # Collect phenomenon data
    run_ids = []
    all_phenomena = set()
    phenomenon_matrix = []
    
    for run in batch_data['runs']:
        if run['status'] != 'completed':
            continue
        
        conv = db.get_conversation(run['conversation_id'])
        if not conv:
            continue
        
        metrics = evaluator.evaluate_template_run(conv, template_type)
        run_ids.append(run['run_id'][-8:])  # Short ID
        all_phenomena.update(metrics.phenomena_detected)
        phenomenon_matrix.append(metrics.phenomena_detected)
    
    if not all_phenomena:
        print("No phenomena detected - skipping heatmap")
        return
    
    # Create binary matrix
    phenomena_list = sorted(all_phenomena)
    matrix = []
    
    for phenomena_set in phenomenon_matrix:
        row = [1 if p in phenomena_set else 0 for p in phenomena_list]
        matrix.append(row)
    
    # Plot
    plt.figure(figsize=(10, max(6, len(run_ids) * 0.5)))
    sns.heatmap(matrix, xticklabels=phenomena_list, yticklabels=run_ids,
                cmap='YlOrRd', cbar_kws={'label': 'Detected'}, linewidths=0.5)
    plt.xlabel('Phenomenon', fontsize=12)
    plt.ylabel('Run ID', fontsize=12)
    plt.title('Phenomenon Detection Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_file = output_path / f"{batch_data['batch_id']}_phenomenon_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_file}")
    plt.close()


def _infer_template_type(template_id: str) -> str:
    """Infer template type"""
    type_mapping = {
        'identity': 'identity_leak_detection',
        'emotional': 'empathy_cascade_study',
        'creativity': 'creative_collaboration',
        'breakdown': 'failure_mode_induction',
    }
    
    for key, value in type_mapping.items():
        if key in template_id.lower():
            return value
    return 'generic'


def main():
    """CLI interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualization.py timeline <batch_result.json>")
        print("  python visualization.py compare <batch1.json> <batch2.json> ...")
        print("  python visualization.py heatmap <batch_result.json>")
        return
    
    command = sys.argv[1]
    
    if command == "timeline":
        plot_identity_leak_timeline(sys.argv[2])
    elif command == "compare":
        plot_success_scores_comparison(sys.argv[2:])
    elif command == "heatmap":
        plot_phenomenon_heatmap(sys.argv[2])
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
