#!/usr/bin/env python3
"""
Batch Runner for Research Templates
Executes multiple template runs with progress tracking, error handling, and statistical analysis
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.template_executor import TemplateExecutor, ExperimentRun
from analysis.quantitative import QuantitativeAnalyzer
from storage.database import Database
from config import Config


@dataclass
class BatchExperimentResult:
    """Results from a batch experiment"""
    batch_id: str
    template_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_runs: int
    completed_runs: int
    failed_runs: int
    total_turns: int
    total_duration_seconds: float
    runs: List[Dict[str, Any]]
    statistics: Optional[Dict[str, Any]] = None
    error_summary: Optional[Dict[str, int]] = None


class BatchRunner:
    """
    Execute research templates in batch with progress tracking
    """
    
    def __init__(self, 
                 output_dir: str = "research_results",
                 parallel: int = 1,
                 enable_analysis: bool = True):
        """
        Initialize batch runner
        
        Args:
            output_dir: Directory for results
            parallel: Number of parallel conversations (use 1 for sequential)
            enable_analysis: Run quantitative analysis on results
        """
        self.executor = TemplateExecutor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.parallel = parallel
        self.enable_analysis = enable_analysis
        self.db = Database(Config.DATABASE_PATH)
    
    def run_template_batch(self, 
                          template_id: str,
                          max_runs: Optional[int] = None,
                          save_progress: bool = True) -> BatchExperimentResult:
        """
        Execute all runs for a template
        
        Args:
            template_id: Template to execute
            max_runs: Limit number of runs (None = all)
            save_progress: Save intermediate results
        
        Returns:
            BatchExperimentResult with all data
        """
        print(f"\n{'=' * 80}")
        print(f"🔬 Batch Execution: {template_id}")
        print(f"{'=' * 80}\n")
        
        # Generate runs
        runs = self.executor.generate_experiment_runs(template_id)
        
        if max_runs:
            runs = runs[:max_runs]
        
        template = self.executor.get_template(template_id)
        
        print(f"📊 Template: {template.get('description')}")
        print(f"🎯 Research Question: {template.get('research_question')}")
        print(f"📈 Total Runs: {len(runs)}")
        print(f"⚙️  Parallel Execution: {self.parallel}")
        print()
        
        # Confirm execution
        estimated_duration = template.get('metadata', {}).get('estimated_duration_minutes', 'unknown')
        print(f"⏱️  Estimated Duration: {estimated_duration} minutes")
        print(f"⚠️  This will consume API tokens")
        print()
        
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return None
        
        # Create batch result
        batch_id = f"{template_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_result = BatchExperimentResult(
            batch_id=batch_id,
            template_id=template_id,
            start_time=datetime.now(),
            end_time=None,
            total_runs=len(runs),
            completed_runs=0,
            failed_runs=0,
            total_turns=0,
            total_duration_seconds=0.0,
            runs=[]
        )
        
        print(f"\n🚀 Starting execution...")
        print(f"{'=' * 80}\n")
        
        start_time = time.time()
        
        # Execute runs
        if self.parallel > 1:
            completed_runs = self._execute_parallel(runs, batch_result)
        else:
            completed_runs = self._execute_sequential(runs, batch_result)
        
        end_time = time.time()
        
        # Update batch result
        batch_result.end_time = datetime.now()
        batch_result.total_duration_seconds = end_time - start_time
        batch_result.runs = [asdict(run) for run in completed_runs]
        
        # Calculate statistics
        batch_result = self._calculate_statistics(batch_result)
        
        # Save results
        if save_progress:
            self._save_batch_result(batch_result)
        
        # Print summary
        self._print_summary(batch_result)
        
        return batch_result
    
    def _execute_sequential(self, runs: List[ExperimentRun], 
                           batch_result: BatchExperimentResult) -> List[ExperimentRun]:
        """Execute runs sequentially"""
        completed = []
        
        for i, run in enumerate(runs, 1):
            print(f"[{i}/{len(runs)}] {run.run_id}")
            
            completed_run = self.executor.execute_run(run, verbose=True)
            completed.append(completed_run)
            
            # Update counters
            if completed_run.status == "completed":
                batch_result.completed_runs += 1
                
                # Get conversation turns
                if completed_run.conversation_id:
                    conv = self.db.get_conversation(completed_run.conversation_id)
                    if conv:
                        batch_result.total_turns += conv.total_turns
            else:
                batch_result.failed_runs += 1
            
            print()
        
        return completed
    
    def _execute_parallel(self, runs: List[ExperimentRun],
                         batch_result: BatchExperimentResult) -> List[ExperimentRun]:
        """Execute runs in parallel"""
        completed = []
        
        print(f"⚡ Running {self.parallel} conversations in parallel\n")
        
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            future_to_run = {
                executor.submit(self.executor.execute_run, run, False): run 
                for run in runs
            }
            
            for i, future in enumerate(as_completed(future_to_run), 1):
                run = future_to_run[future]
                try:
                    completed_run = future.result()
                    completed.append(completed_run)
                    
                    # Update counters
                    if completed_run.status == "completed":
                        batch_result.completed_runs += 1
                        
                        if completed_run.conversation_id:
                            conv = self.db.get_conversation(completed_run.conversation_id)
                            if conv:
                                batch_result.total_turns += conv.total_turns
                        
                        print(f"[{i}/{len(runs)}] ✅ {completed_run.run_id}")
                    else:
                        batch_result.failed_runs += 1
                        print(f"[{i}/{len(runs)}] ❌ {completed_run.run_id}: {completed_run.error}")
                
                except Exception as e:
                    print(f"[{i}/{len(runs)}] ❌ {run.run_id}: {e}")
                    run.status = "error"
                    run.error = str(e)
                    completed.append(run)
                    batch_result.failed_runs += 1
        
        return completed
    
    def _calculate_statistics(self, batch_result: BatchExperimentResult) -> BatchExperimentResult:
        """Calculate statistical summaries"""
        completed_runs = [r for r in batch_result.runs if r['status'] == 'completed']
        
        if not completed_runs:
            batch_result.statistics = {"error": "No completed runs"}
            return batch_result
        
        # Collect conversation data
        turn_counts = []
        conversations = []
        
        for run in completed_runs:
            if run.get('conversation_id'):
                conv = self.db.get_conversation(run['conversation_id'])
                if conv:
                    turn_counts.append(conv.total_turns)
                    conversations.append(conv)
        
        if not turn_counts:
            batch_result.statistics = {"error": "No conversation data"}
            return batch_result
        
        # Basic statistics
        stats = {
            "completion_rate": len(completed_runs) / batch_result.total_runs,
            "turn_statistics": {
                "mean": statistics.mean(turn_counts),
                "median": statistics.median(turn_counts),
                "stdev": statistics.stdev(turn_counts) if len(turn_counts) > 1 else 0,
                "min": min(turn_counts),
                "max": max(turn_counts),
                "total": sum(turn_counts)
            },
            "timing": {
                "total_duration_seconds": batch_result.total_duration_seconds,
                "avg_seconds_per_run": batch_result.total_duration_seconds / batch_result.total_runs,
                "total_turns": sum(turn_counts),
                "avg_seconds_per_turn": batch_result.total_duration_seconds / sum(turn_counts) if sum(turn_counts) > 0 else 0
            }
        }
        
        # Run quantitative analysis on sample conversations
        if self.enable_analysis and conversations:
            print(f"\n📊 Running quantitative analysis on {len(conversations)} conversations...")
            
            analysis_results = []
            for conv in conversations[:10]:  # Analyze first 10
                try:
                    analyzer = QuantitativeAnalyzer(conv.messages)
                    report = analyzer.generate_full_report()
                    analysis_results.append(report)
                except Exception as e:
                    print(f"  ⚠️  Analysis error for conv {conv.id}: {e}")
            
            if analysis_results:
                # Aggregate metrics
                stats["analysis_aggregates"] = self._aggregate_analysis_results(analysis_results)
        
        # Error summary
        errors = {}
        for run in batch_result.runs:
            if run['status'] == 'error' and run.get('error'):
                error_type = run['error'].split(':')[0]
                errors[error_type] = errors.get(error_type, 0) + 1
        
        batch_result.error_summary = errors
        batch_result.statistics = stats
        
        return batch_result
    
    def _aggregate_analysis_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate multiple analysis results"""
        aggregates = {}
        
        # Conversation dynamics
        balance_ratios = [r['conversation_dynamics']['turn_taking']['turn_balance_ratio'] 
                         for r in results if 'conversation_dynamics' in r]
        entropies = [r['conversation_dynamics']['information_flow']['shannon_entropy']
                    for r in results if 'conversation_dynamics' in r]
        
        aggregates["turn_balance"] = {
            "mean": statistics.mean(balance_ratios) if balance_ratios else 0,
            "stdev": statistics.stdev(balance_ratios) if len(balance_ratios) > 1 else 0
        }
        
        aggregates["information_entropy"] = {
            "mean": statistics.mean(entropies) if entropies else 0,
            "stdev": statistics.stdev(entropies) if len(entropies) > 1 else 0
        }
        
        # Linguistic complexity
        grade_levels = [r['linguistic_analysis']['readability']['overall_grade_level']
                       for r in results if 'linguistic_analysis' in r]
        
        aggregates["readability"] = {
            "mean_grade_level": statistics.mean(grade_levels) if grade_levels else 0,
            "stdev": statistics.stdev(grade_levels) if len(grade_levels) > 1 else 0
        }
        
        return aggregates
    
    def _save_batch_result(self, batch_result: BatchExperimentResult):
        """Save batch result to file"""
        result_file = self.output_dir / f"{batch_result.batch_id}_results.json"
        
        # Convert to JSON-serializable format
        result_dict = asdict(batch_result)
        result_dict['start_time'] = batch_result.start_time.isoformat()
        result_dict['end_time'] = batch_result.end_time.isoformat() if batch_result.end_time else None
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {result_file}")
    
    def _print_summary(self, batch_result: BatchExperimentResult):
        """Print batch execution summary"""
        print(f"\n{'=' * 80}")
        print(f"📊 BATCH EXECUTION SUMMARY")
        print(f"{'=' * 80}\n")
        
        print(f"Batch ID: {batch_result.batch_id}")
        print(f"Template: {batch_result.template_id}")
        print(f"Duration: {batch_result.total_duration_seconds:.1f} seconds")
        print()
        
        print(f"📈 Execution Results:")
        print(f"  Total Runs: {batch_result.total_runs}")
        print(f"  Completed: {batch_result.completed_runs} ({batch_result.completed_runs/batch_result.total_runs*100:.1f}%)")
        print(f"  Failed: {batch_result.failed_runs}")
        print(f"  Total Turns: {batch_result.total_turns}")
        print()
        
        if batch_result.statistics:
            stats = batch_result.statistics
            
            if "turn_statistics" in stats:
                ts = stats["turn_statistics"]
                print(f"🔢 Turn Statistics:")
                print(f"  Mean: {ts['mean']:.1f}")
                print(f"  Median: {ts['median']:.1f}")
                print(f"  Std Dev: {ts['stdev']:.2f}")
                print(f"  Range: {ts['min']} - {ts['max']}")
                print()
            
            if "timing" in stats:
                timing = stats["timing"]
                print(f"⏱️  Timing:")
                print(f"  Avg seconds/run: {timing['avg_seconds_per_run']:.1f}")
                print(f"  Avg seconds/turn: {timing['avg_seconds_per_turn']:.1f}")
                print()
            
            if "analysis_aggregates" in stats:
                agg = stats["analysis_aggregates"]
                print(f"📊 Analysis Aggregates:")
                if "turn_balance" in agg:
                    print(f"  Turn Balance: {agg['turn_balance']['mean']:.2f} ± {agg['turn_balance']['stdev']:.2f}")
                if "information_entropy" in agg:
                    print(f"  Information Entropy: {agg['information_entropy']['mean']:.2f} ± {agg['information_entropy']['stdev']:.2f}")
                if "readability" in agg:
                    print(f"  Mean Grade Level: {agg['readability']['mean_grade_level']:.1f}")
                print()
        
        if batch_result.error_summary:
            print(f"❌ Errors:")
            for error_type, count in batch_result.error_summary.items():
                print(f"  {error_type}: {count}")
            print()
        
        print(f"{'=' * 80}\n")
    
    def run_multiple_templates(self, 
                              template_ids: List[str],
                              max_runs_per_template: Optional[int] = None) -> List[BatchExperimentResult]:
        """
        Run multiple templates sequentially
        
        Args:
            template_ids: List of template IDs to execute
            max_runs_per_template: Limit runs per template
        
        Returns:
            List of BatchExperimentResults
        """
        results = []
        
        print(f"\n{'=' * 80}")
        print(f"🔬 MULTI-TEMPLATE BATCH EXECUTION")
        print(f"{'=' * 80}\n")
        print(f"Templates to execute: {len(template_ids)}")
        print(f"  {', '.join(template_ids)}")
        print()
        
        for i, template_id in enumerate(template_ids, 1):
            print(f"\n[{i}/{len(template_ids)}] Executing template: {template_id}")
            print(f"{'=' * 80}")
            
            try:
                result = self.run_template_batch(template_id, max_runs=max_runs_per_template)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ Error executing {template_id}: {e}")
                continue
        
        # Generate comparative report
        if results:
            self._generate_comparative_report(results)
        
        return results
    
    def _generate_comparative_report(self, results: List[BatchExperimentResult]):
        """Generate comparative report across multiple templates"""
        report_path = self.output_dir / f"comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# Multi-Template Comparative Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("## Summary Table\n\n")
            f.write("| Template | Runs | Completed | Failed | Avg Turns | Duration (s) |\n")
            f.write("|----------|------|-----------|--------|-----------|-------------|\n")
            
            for result in results:
                avg_turns = result.total_turns / result.completed_runs if result.completed_runs > 0 else 0
                f.write(f"| {result.template_id} | {result.total_runs} | "
                       f"{result.completed_runs} | {result.failed_runs} | "
                       f"{avg_turns:.1f} | {result.total_duration_seconds:.1f} |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            for result in results:
                f.write(f"### {result.template_id}\n\n")
                f.write(f"**Batch ID:** {result.batch_id}\n\n")
                
                if result.statistics:
                    f.write("**Statistics:**\n")
                    f.write(f"```json\n{json.dumps(result.statistics, indent=2)}\n```\n\n")
        
        print(f"\n📄 Comparative report saved to: {report_path}")


def main():
    """CLI interface for batch runner"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python batch_runner.py <command> [args]")
        print("\nCommands:")
        print("  run <template_id> [--max-runs N] [--parallel N]")
        print("  run-multiple <template_id1,template_id2,...> [--max-runs N]")
        print("  run-priority <priority> [--max-runs N]")
        return
    
    command = sys.argv[1]
    
    # Parse common args
    max_runs = None
    parallel = 1
    
    if "--max-runs" in sys.argv:
        idx = sys.argv.index("--max-runs")
        max_runs = int(sys.argv[idx + 1])
    
    if "--parallel" in sys.argv:
        idx = sys.argv.index("--parallel")
        parallel = int(sys.argv[idx + 1])
    
    runner = BatchRunner(parallel=parallel)
    
    if command == "run":
        if len(sys.argv) < 3:
            print("Usage: python batch_runner.py run <template_id> [--max-runs N] [--parallel N]")
            return
        
        template_id = sys.argv[2]
        runner.run_template_batch(template_id, max_runs=max_runs)
    
    elif command == "run-multiple":
        if len(sys.argv) < 3:
            print("Usage: python batch_runner.py run-multiple <id1,id2,...> [--max-runs N]")
            return
        
        template_ids = sys.argv[2].split(',')
        runner.run_multiple_templates(template_ids, max_runs_per_template=max_runs)
    
    elif command == "run-priority":
        if len(sys.argv) < 3:
            print("Usage: python batch_runner.py run-priority <critical|high|medium> [--max-runs N]")
            return
        
        priority = sys.argv[2]
        
        # Get templates by priority
        templates = runner.executor.list_templates()
        priority_templates = [t['template_id'] for t in templates 
                            if t.get('priority') == priority]
        
        print(f"Found {len(priority_templates)} templates with priority '{priority}'")
        runner.run_multiple_templates(priority_templates, max_runs_per_template=max_runs)


if __name__ == "__main__":
    main()
