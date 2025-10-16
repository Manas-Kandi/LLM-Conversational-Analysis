#!/usr/bin/env python3
"""
Factorial Experiment Analysis for Research Paper
Analyzes the full factorial design: Prompt Type × Temperature × Replicates
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import statistics


class FactorialAnalyzer:
    """Analyzes factorial experimental data for paper"""
    
    def __init__(self, 
                 factorial_results_path: str = "research_results/full_factorial_20251008_191343_results.json",
                 conversations_dir: str = "conversations_json"):
        self.factorial_path = Path(factorial_results_path)
        self.conversations_dir = Path(conversations_dir)
        self.experimental_data = None
        self.conversations = {}
        
        self.load_factorial_data()
        
    def load_factorial_data(self):
        """Load the factorial experiment results"""
        print(f"Loading factorial experiment data from {self.factorial_path}...")
        
        with open(self.factorial_path, 'r') as f:
            self.experimental_data = json.load(f)
        
        print(f"Loaded {len(self.experimental_data['runs'])} experimental runs")
        print(f"Batch ID: {self.experimental_data['batch_id']}")
        print(f"Total planned runs: {self.experimental_data['total_runs']}")
    
    def load_conversation(self, conv_id: int) -> Dict:
        """Load a specific conversation JSON file"""
        if conv_id in self.conversations:
            return self.conversations[conv_id]
        
        # Find the conversation file
        conv_files = list(self.conversations_dir.glob(f"conv_{conv_id}_*.json"))
        
        if not conv_files:
            print(f"Warning: Conversation {conv_id} not found")
            return None
        
        with open(conv_files[0], 'r') as f:
            conv = json.load(f)
            self.conversations[conv_id] = conv
            return conv
    
    def analyze_factorial_structure(self) -> Dict[str, Any]:
        """Analyze the experimental design structure"""
        runs = self.experimental_data['runs']
        
        # Extract factors
        prompt_types = set()
        temperatures = set()
        replicates = defaultdict(int)
        
        for run in runs:
            prompt_types.add(run['prompt_type'])
            temperatures.add(run['temperature'])
            condition = f"{run['prompt_type']}_T{int(run['temperature']*100):03d}"
            replicates[condition] += 1
        
        return {
            "design": "Full Factorial",
            "factors": {
                "Prompt Type": sorted(list(prompt_types)),
                "Temperature": sorted(list(temperatures))
            },
            "levels": {
                "Prompt Type": len(prompt_types),
                "Temperature": len(temperatures)
            },
            "total_conditions": len(prompt_types) * len(temperatures),
            "replicates_per_condition": dict(replicates),
            "target_replicates": 5,
            "total_observations": len(runs),
            "completed_runs": sum(1 for r in runs if r['status'] == 'completed')
        }
    
    def analyze_temperature_effects(self) -> Dict[str, Any]:
        """Analyze main effect of temperature"""
        runs = self.experimental_data['runs']
        
        # Group by temperature
        temp_data = defaultdict(list)
        
        for run in runs:
            if run['status'] == 'completed':
                temp = run['temperature']
                conv_id = run['conversation_id']
                
                # Load conversation to get metrics
                conv = self.load_conversation(conv_id)
                if conv:
                    # Extract basic metrics
                    turns = conv['metadata'].get('total_turns', 0)
                    messages = conv.get('messages', [])
                    
                    # Check for breakdown indicators
                    has_gibberish = self._check_gibberish(messages)
                    high_ai_mentions = self._check_ai_mentions(messages)
                    
                    temp_data[temp].append({
                        'turns': turns,
                        'gibberish': has_gibberish,
                        'ai_mentions': high_ai_mentions,
                        'conv_id': conv_id
                    })
        
        # Calculate statistics per temperature
        temp_stats = {}
        for temp in sorted(temp_data.keys()):
            convs = temp_data[temp]
            turns_list = [c['turns'] for c in convs]
            gibberish_count = sum(1 for c in convs if c['gibberish'])
            ai_mention_count = sum(1 for c in convs if c['ai_mentions'])
            
            temp_stats[temp] = {
                'n': len(convs),
                'mean_turns': statistics.mean(turns_list) if turns_list else 0,
                'median_turns': statistics.median(turns_list) if turns_list else 0,
                'stdev_turns': statistics.stdev(turns_list) if len(turns_list) > 1 else 0,
                'min_turns': min(turns_list) if turns_list else 0,
                'max_turns': max(turns_list) if turns_list else 0,
                'gibberish_rate': (gibberish_count / len(convs) * 100) if convs else 0,
                'ai_mention_rate': (ai_mention_count / len(convs) * 100) if convs else 0
            }
        
        return temp_stats
    
    def analyze_prompt_effects(self) -> Dict[str, Any]:
        """Analyze main effect of prompt type"""
        runs = self.experimental_data['runs']
        
        # Group by prompt type
        prompt_data = defaultdict(list)
        
        for run in runs:
            if run['status'] == 'completed':
                prompt_type = run['prompt_type']
                conv_id = run['conversation_id']
                
                conv = self.load_conversation(conv_id)
                if conv:
                    turns = conv['metadata'].get('total_turns', 0)
                    messages = conv.get('messages', [])
                    
                    has_gibberish = self._check_gibberish(messages)
                    
                    prompt_data[prompt_type].append({
                        'turns': turns,
                        'gibberish': has_gibberish,
                        'conv_id': conv_id,
                        'temp': run['temperature']
                    })
        
        # Calculate statistics per prompt type
        prompt_stats = {}
        for ptype in sorted(prompt_data.keys()):
            convs = prompt_data[ptype]
            turns_list = [c['turns'] for c in convs]
            gibberish_count = sum(1 for c in convs if c['gibberish'])
            
            prompt_stats[ptype] = {
                'n': len(convs),
                'mean_turns': statistics.mean(turns_list) if turns_list else 0,
                'median_turns': statistics.median(turns_list) if turns_list else 0,
                'stdev_turns': statistics.stdev(turns_list) if len(turns_list) > 1 else 0,
                'gibberish_rate': (gibberish_count / len(convs) * 100) if convs else 0,
                'temp_range': f"{min(c['temp'] for c in convs):.1f}-{max(c['temp'] for c in convs):.1f}"
            }
        
        return prompt_stats
    
    def analyze_interactions(self) -> Dict[str, Any]:
        """Analyze Prompt Type × Temperature interactions"""
        runs = self.experimental_data['runs']
        
        # Create 2D array of conditions
        interaction_data = defaultdict(lambda: defaultdict(list))
        
        for run in runs:
            if run['status'] == 'completed':
                ptype = run['prompt_type']
                temp = run['temperature']
                conv_id = run['conversation_id']
                
                conv = self.load_conversation(conv_id)
                if conv:
                    turns = conv['metadata'].get('total_turns', 0)
                    messages = conv.get('messages', [])
                    has_gibberish = self._check_gibberish(messages)
                    
                    interaction_data[ptype][temp].append({
                        'turns': turns,
                        'gibberish': has_gibberish
                    })
        
        # Calculate means for interaction plot
        interaction_matrix = {}
        for ptype in sorted(interaction_data.keys()):
            interaction_matrix[ptype] = {}
            for temp in sorted(interaction_data[ptype].keys()):
                convs = interaction_data[ptype][temp]
                turns_list = [c['turns'] for c in convs]
                gibberish_count = sum(1 for c in convs if c['gibberish'])
                
                interaction_matrix[ptype][temp] = {
                    'n': len(convs),
                    'mean_turns': statistics.mean(turns_list) if turns_list else 0,
                    'gibberish_rate': (gibberish_count / len(convs) * 100) if convs else 0
                }
        
        return interaction_matrix
    
    def _check_gibberish(self, messages: List[Dict]) -> bool:
        """Check if conversation has gibberish (token cascade failure)"""
        if not messages:
            return False
        
        # Check last 10 messages
        for msg in messages[-10:]:
            content = msg.get('content', '')
            if len(content) > 0:
                non_alpha = sum(1 for c in content if not c.isalpha() and not c.isspace())
                ratio = non_alpha / len(content)
                if ratio > 0.3:  # More than 30% non-alphabetic
                    return True
        
        return False
    
    def _check_ai_mentions(self, messages: List[Dict]) -> bool:
        """Check if conversation has high AI self-reference"""
        if not messages:
            return False
        
        ai_mentions = sum(
            1 for msg in messages 
            if 'conversational AI' in msg.get('content', '').lower() or 
               'as an AI' in msg.get('content', '').lower()
        )
        
        return ai_mentions > len(messages) * 0.3  # More than 30% mention AI
    
    def generate_paper_tables(self) -> str:
        """Generate markdown tables for paper"""
        output = []
        
        output.append("# Factorial Experiment Analysis Results\n\n")
        output.append("## Experimental Design\n\n")
        
        # Design structure
        structure = self.analyze_factorial_structure()
        output.append(f"- **Design**: {structure['design']}\n")
        output.append(f"- **Factors**: {len(structure['factors'])}\n")
        output.append(f"- **Total Conditions**: {structure['total_conditions']}\n")
        output.append(f"- **Target Replicates**: {structure['target_replicates']}\n")
        output.append(f"- **Total Observations**: {structure['total_observations']}\n")
        output.append(f"- **Completed Runs**: {structure['completed_runs']}\n\n")
        
        output.append("### Factor Levels\n\n")
        output.append(f"- **Prompt Types**: {', '.join(structure['factors']['Prompt Type'])}\n")
        output.append(f"- **Temperatures**: {', '.join(map(str, structure['factors']['Temperature']))}\n\n")
        
        # Temperature main effects
        output.append("## Main Effect: Temperature\n\n")
        output.append("| Temperature | N | Mean Turns | SD | Gibberish Rate (%) |\n")
        output.append("|-------------|---|------------|----|--------------------|------|\n")
        
        temp_stats = self.analyze_temperature_effects()
        for temp in sorted(temp_stats.keys()):
            stats = temp_stats[temp]
            output.append(
                f"| {temp:.1f} | {stats['n']} | {stats['mean_turns']:.2f} | "
                f"{stats['stdev_turns']:.2f} | {stats['gibberish_rate']:.1f}% |\n"
            )
        
        output.append("\n")
        
        # Prompt type main effects
        output.append("## Main Effect: Prompt Type\n\n")
        output.append("| Prompt Type | N | Mean Turns | SD | Gibberish Rate (%) |\n")
        output.append("|-------------|---|------------|----|--------------------|------|\n")
        
        prompt_stats = self.analyze_prompt_effects()
        for ptype in sorted(prompt_stats.keys()):
            stats = prompt_stats[ptype]
            output.append(
                f"| {ptype} | {stats['n']} | {stats['mean_turns']:.2f} | "
                f"{stats['stdev_turns']:.2f} | {stats['gibberish_rate']:.1f}% |\n"
            )
        
        output.append("\n")
        
        # Interaction effects
        output.append("## Interaction: Prompt Type × Temperature\n\n")
        output.append("Mean turns completed by condition:\n\n")
        
        interaction_matrix = self.analyze_interactions()
        
        # Get all temps
        all_temps = set()
        for ptype_data in interaction_matrix.values():
            all_temps.update(ptype_data.keys())
        all_temps = sorted(all_temps)
        
        # Table header
        output.append("| Prompt | " + " | ".join(f"T={t:.1f}" for t in all_temps) + " |\n")
        output.append("|--------|" + "|".join("-------" for _ in all_temps) + "|\n")
        
        # Table rows
        for ptype in sorted(interaction_matrix.keys()):
            row = [ptype]
            for temp in all_temps:
                if temp in interaction_matrix[ptype]:
                    mean_turns = interaction_matrix[ptype][temp]['mean_turns']
                    row.append(f"{mean_turns:.1f}")
                else:
                    row.append("--")
            output.append("| " + " | ".join(row) + " |\n")
        
        output.append("\n")
        
        return "".join(output)
    
    def run_full_analysis(self):
        """Run complete analysis and save results"""
        print("=" * 60)
        print("FACTORIAL EXPERIMENT ANALYSIS")
        print("=" * 60)
        print()
        
        # Generate tables
        tables = self.generate_paper_tables()
        
        # Save to file
        output_file = "factorial_analysis_report.md"
        with open(output_file, 'w') as f:
            f.write(tables)
        
        print(f"✓ Analysis complete!")
        print(f"✓ Report saved to: {output_file}")
        print()
        
        # Print summary
        structure = self.analyze_factorial_structure()
        print("SUMMARY:")
        print(f"  - Experimental runs: {structure['completed_runs']}/{structure['total_observations']}")
        print(f"  - Prompt types: {structure['levels']['Prompt Type']}")
        print(f"  - Temperature levels: {structure['levels']['Temperature']}")
        print(f"  - Total conditions: {structure['total_conditions']}")
        print()
        
        temp_stats = self.analyze_temperature_effects()
        print("KEY FINDINGS:")
        print(f"  - Temperature range: {min(temp_stats.keys()):.1f} - {max(temp_stats.keys()):.1f}")
        
        # Find temperature with highest gibberish rate
        max_gibberish_temp = max(temp_stats.items(), key=lambda x: x[1]['gibberish_rate'])
        print(f"  - Highest breakdown rate: T={max_gibberish_temp[0]:.1f} ({max_gibberish_temp[1]['gibberish_rate']:.1f}%)")
        
        # Find temperature with best performance
        max_turns_temp = max(temp_stats.items(), key=lambda x: x[1]['mean_turns'])
        print(f"  - Best performance: T={max_turns_temp[0]:.1f} (avg {max_turns_temp[1]['mean_turns']:.1f} turns)")


def main():
    """Main entry point"""
    analyzer = FactorialAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
