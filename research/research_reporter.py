#!/usr/bin/env python3
"""
Research Reporter for AA Microscope
Generate comprehensive research reports from template experiments
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import statistics
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.database import Database
from research.template_metrics import TemplateEvaluator
from config import Config


class ResearchReporter:
    """Generate comprehensive research reports"""
    
    def __init__(self, output_dir: str = "research_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.db = Database(Config.DATABASE_PATH)
        self.evaluator = TemplateEvaluator()
    
    def generate_template_report(self, 
                                batch_result_file: str,
                                include_detailed_analysis: bool = True) -> Path:
        """
        Generate detailed report for a single template batch
        
        Args:
            batch_result_file: Path to batch result JSON
            include_detailed_analysis: Include per-conversation analysis
        
        Returns:
            Path to generated report
        """
        # Load batch results
        with open(batch_result_file, 'r') as f:
            batch_data = json.load(f)
        
        template_id = batch_data['template_id']
        batch_id = batch_data['batch_id']
        
        report_path = self.output_dir / f"{batch_id}_report.md"
        
        with open(report_path, 'w') as f:
            self._write_header(f, batch_data)
            self._write_executive_summary(f, batch_data)
            self._write_statistical_overview(f, batch_data)
            
            if include_detailed_analysis:
                self._write_detailed_analysis(f, batch_data)
            
            self._write_phenomena_summary(f, batch_data)
            self._write_recommendations(f, batch_data)
        
        print(f"📄 Report generated: {report_path}")
        return report_path
    
    def _write_header(self, f, batch_data: Dict):
        """Write report header"""
        f.write(f"# Research Report: {batch_data['template_id']}\n\n")
        f.write(f"**Batch ID:** {batch_data['batch_id']}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
    
    def _write_executive_summary(self, f, batch_data: Dict):
        """Write executive summary"""
        f.write("## Executive Summary\n\n")
        
        completion_rate = batch_data['completed_runs'] / batch_data['total_runs'] * 100
        
        f.write(f"- **Total Runs:** {batch_data['total_runs']}\n")
        f.write(f"- **Completed:** {batch_data['completed_runs']} ({completion_rate:.1f}%)\n")
        f.write(f"- **Failed:** {batch_data['failed_runs']}\n")
        f.write(f"- **Total Turns:** {batch_data['total_turns']}\n")
        f.write(f"- **Duration:** {batch_data['total_duration_seconds']:.1f} seconds\n\n")
        
        if batch_data.get('statistics'):
            stats = batch_data['statistics']
            if 'turn_statistics' in stats:
                ts = stats['turn_statistics']
                f.write(f"- **Avg Turns/Conversation:** {ts['mean']:.1f} ± {ts['stdev']:.1f}\n")
        
        f.write("\n")
    
    def _write_statistical_overview(self, f, batch_data: Dict):
        """Write statistical overview"""
        f.write("## Statistical Overview\n\n")
        
        stats = batch_data.get('statistics', {})
        
        if 'turn_statistics' in stats:
            ts = stats['turn_statistics']
            f.write("### Turn Statistics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Mean | {ts['mean']:.2f} |\n")
            f.write(f"| Median | {ts['median']:.2f} |\n")
            f.write(f"| Std Dev | {ts['stdev']:.2f} |\n")
            f.write(f"| Min | {ts['min']} |\n")
            f.write(f"| Max | {ts['max']} |\n")
            f.write(f"| Total | {ts['total']} |\n\n")
        
        if 'analysis_aggregates' in stats:
            agg = stats['analysis_aggregates']
            f.write("### Analysis Aggregates\n\n")
            
            if 'turn_balance' in agg:
                f.write(f"**Turn Balance:** {agg['turn_balance']['mean']:.3f} ± {agg['turn_balance']['stdev']:.3f}\n\n")
            
            if 'information_entropy' in agg:
                f.write(f"**Information Entropy:** {agg['information_entropy']['mean']:.2f} ± {agg['information_entropy']['stdev']:.2f}\n\n")
            
            if 'readability' in agg:
                f.write(f"**Mean Grade Level:** {agg['readability']['mean_grade_level']:.1f}\n\n")
        
        if batch_data.get('error_summary'):
            f.write("### Error Summary\n\n")
            for error_type, count in batch_data['error_summary'].items():
                f.write(f"- **{error_type}:** {count} occurrences\n")
            f.write("\n")
    
    def _write_detailed_analysis(self, f, batch_data: Dict):
        """Write per-conversation detailed analysis"""
        f.write("## Detailed Analysis\n\n")
        
        completed_runs = [r for r in batch_data['runs'] if r['status'] == 'completed']
        
        if not completed_runs:
            f.write("No completed runs to analyze.\n\n")
            return
        
        f.write(f"Analyzing {len(completed_runs)} completed conversations...\n\n")
        
        # Analyze each conversation
        template_type = self._infer_template_type(batch_data['template_id'])
        
        for i, run in enumerate(completed_runs[:10], 1):  # Limit to first 10
            conv_id = run.get('conversation_id')
            if not conv_id:
                continue
            
            conv = self.db.get_conversation(conv_id)
            if not conv:
                continue
            
            f.write(f"### Conversation {i}: {run['run_id']}\n\n")
            
            # Evaluate with template-specific metrics
            metrics = self.evaluator.evaluate_template_run(conv, template_type)
            
            f.write(f"**Success Score:** {metrics.success_score:.2f}\n\n")
            
            if metrics.phenomena_detected:
                f.write(f"**Phenomena Detected:** {', '.join(metrics.phenomena_detected)}\n\n")
            
            if metrics.notes:
                f.write("**Notes:**\n")
                for note in metrics.notes:
                    f.write(f"- {note}\n")
                f.write("\n")
            
            # Key metrics
            f.write("**Key Metrics:**\n")
            f.write(f"```json\n{json.dumps(metrics.metrics, indent=2)}\n```\n\n")
    
    def _write_phenomena_summary(self, f, batch_data: Dict):
        """Summarize emergent phenomena across runs"""
        f.write("## Emergent Phenomena Summary\n\n")
        
        completed_runs = [r for r in batch_data['runs'] if r['status'] == 'completed']
        template_type = self._infer_template_type(batch_data['template_id'])
        
        all_phenomena = []
        success_scores = []
        
        for run in completed_runs:
            conv_id = run.get('conversation_id')
            if not conv_id:
                continue
            
            conv = self.db.get_conversation(conv_id)
            if not conv:
                continue
            
            metrics = self.evaluator.evaluate_template_run(conv, template_type)
            all_phenomena.extend(metrics.phenomena_detected)
            success_scores.append(metrics.success_score)
        
        if all_phenomena:
            from collections import Counter
            phenomenon_counts = Counter(all_phenomena)
            
            f.write("| Phenomenon | Frequency | Percentage |\n")
            f.write("|------------|-----------|------------|\n")
            
            for phenomenon, count in phenomenon_counts.most_common():
                pct = count / len(completed_runs) * 100
                f.write(f"| {phenomenon} | {count} | {pct:.1f}% |\n")
            
            f.write("\n")
        
        if success_scores:
            f.write(f"**Overall Success Rate:** {statistics.mean(success_scores):.2f}\n\n")
    
    def _write_recommendations(self, f, batch_data: Dict):
        """Write research recommendations"""
        f.write("## Recommendations\n\n")
        
        completion_rate = batch_data['completed_runs'] / batch_data['total_runs']
        
        if completion_rate < 0.8:
            f.write("### ⚠️ Low Completion Rate\n\n")
            f.write(f"Only {completion_rate*100:.1f}% of runs completed successfully. ")
            f.write("Consider:\n")
            f.write("- Reviewing error logs for common failure patterns\n")
            f.write("- Adjusting parameters (temperature, max_turns, etc.)\n")
            f.write("- Checking API connectivity and rate limits\n\n")
        
        stats = batch_data.get('statistics', {})
        if 'turn_statistics' in stats:
            ts = stats['turn_statistics']
            if ts['stdev'] > ts['mean'] * 0.5:
                f.write("### 📊 High Variance in Turn Counts\n\n")
                f.write(f"Turn count std dev ({ts['stdev']:.1f}) is high relative to mean ({ts['mean']:.1f}). ")
                f.write("This suggests:\n")
                f.write("- Inconsistent conversation dynamics\n")
                f.write("- May need to increase sample size for statistical power\n")
                f.write("- Consider analyzing outliers separately\n\n")
        
        f.write("### 🔬 Next Steps\n\n")
        f.write("1. Review detailed analysis for interesting edge cases\n")
        f.write("2. Run comparative analysis with other templates\n")
        f.write("3. Consider follow-up experiments based on emergent phenomena\n")
        f.write("4. Document findings in research notes\n\n")
    
    def _infer_template_type(self, template_id: str) -> str:
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
    
    def generate_comparative_report(self, 
                                   batch_result_files: List[str]) -> Path:
        """
        Generate comparative report across multiple template batches
        
        Args:
            batch_result_files: List of batch result JSON files
        
        Returns:
            Path to generated report
        """
        report_path = self.output_dir / f"comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # Load all batch results
        all_batches = []
        for file in batch_result_files:
            with open(file, 'r') as f:
                all_batches.append(json.load(f))
        
        with open(report_path, 'w') as f:
            f.write("# Comparative Research Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Batches Compared:** {len(all_batches)}\n\n")
            f.write("---\n\n")
            
            # Comparison table
            f.write("## Batch Comparison\n\n")
            f.write("| Template | Runs | Completed | Avg Turns | Success Rate | Duration (s) |\n")
            f.write("|----------|------|-----------|-----------|--------------|-------------|\n")
            
            for batch in all_batches:
                template_id = batch['template_id']
                total = batch['total_runs']
                completed = batch['completed_runs']
                success_rate = completed / total * 100 if total > 0 else 0
                
                avg_turns = 0
                if batch.get('statistics') and 'turn_statistics' in batch['statistics']:
                    avg_turns = batch['statistics']['turn_statistics']['mean']
                
                f.write(f"| {template_id} | {total} | {completed} | "
                       f"{avg_turns:.1f} | {success_rate:.1f}% | "
                       f"{batch['total_duration_seconds']:.0f} |\n")
            
            f.write("\n")
            
            # Statistical comparisons
            f.write("## Statistical Comparisons\n\n")
            
            for batch in all_batches:
                f.write(f"### {batch['template_id']}\n\n")
                
                if batch.get('statistics'):
                    stats = batch['statistics']
                    
                    if 'analysis_aggregates' in stats:
                        agg = stats['analysis_aggregates']
                        f.write("**Analysis Metrics:**\n")
                        f.write(f"```json\n{json.dumps(agg, indent=2)}\n```\n\n")
        
        print(f"📄 Comparative report generated: {report_path}")
        return report_path


def main():
    """CLI interface for research reporter"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python research_reporter.py <command> [args]")
        print("\nCommands:")
        print("  report <batch_result.json>                 - Generate single batch report")
        print("  compare <result1.json> <result2.json> ...  - Generate comparative report")
        return
    
    command = sys.argv[1]
    reporter = ResearchReporter()
    
    if command == "report":
        if len(sys.argv) < 3:
            print("Usage: python research_reporter.py report <batch_result.json>")
            return
        
        batch_file = sys.argv[2]
        reporter.generate_template_report(batch_file)
    
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Usage: python research_reporter.py compare <result1.json> <result2.json> ...")
            return
        
        batch_files = sys.argv[2:]
        reporter.generate_comparative_report(batch_files)
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
