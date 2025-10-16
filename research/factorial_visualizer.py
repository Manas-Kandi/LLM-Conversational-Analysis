#!/usr/bin/env python3
"""
Factorial Experiment Visualization
Generates plots for 4×6 factorial design analysis
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.database import Database
from research.template_metrics import TemplateEvaluator
from config import Config


class FactorialVisualizer:
    """Generate visualizations for factorial experiment"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.db = Database(Config.DATABASE_PATH)
        self.evaluator = TemplateEvaluator()
        
        with open(results_file, 'r') as f:
            self.batch_data = json.load(f)
        
        self.df = None
        self.output_dir = Path("research_results/plots/factorial")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_data()
    
    def _load_data(self):
        """Load all conversation data"""
        rows = []
        
        for run in self.batch_data['runs']:
            if run['status'] != 'completed':
                continue
            
            conv = self.db.get_conversation(run['conversation_id'])
            if not conv:
                continue
            
            metrics = self.evaluator.identity_detector.detect_identity_leak(conv.messages)
            
            row = {
                'condition_code': run['condition_code'],
                'prompt_type': run['prompt_type'],
                'temperature': run['temperature'],
                'replicate': run['replicate'],
                'leak_rate': metrics['leak_rate'],
                'first_leak_turn': metrics.get('first_leak_turn', 30),
                'ai_references': len(metrics.get('ai_keyword_locations', [])),
            }
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        print(f"📊 Loaded {len(self.df)} conversations for visualization")
    
    def plot_interaction(self):
        """Create interaction plot: Temperature × System Prompt"""
        print("Creating interaction plot...")
        
        # Calculate means and standard errors
        interaction_data = self.df.groupby(['prompt_type', 'temperature'])['leak_rate'].agg(['mean', 'sem']).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot lines for each prompt type
        prompt_labels = {'N': 'Neutral', 'S': 'Stealth', 'H': 'Honest', 'X': 'None'}
        colors = {'N': '#3498db', 'S': '#2ecc71', 'H': '#e74c3c', 'X': '#95a5a6'}
        markers = {'N': 'o', 'S': 's', 'H': '^', 'X': 'd'}
        
        for prompt in interaction_data['prompt_type'].unique():
            subset = interaction_data[interaction_data['prompt_type'] == prompt]
            ax.plot(subset['temperature'], subset['mean'], 
                   marker=markers[prompt], markersize=10, linewidth=2.5,
                   label=prompt_labels[prompt], color=colors[prompt])
            
            # Add error bars
            ax.fill_between(subset['temperature'], 
                           subset['mean'] - 1.96*subset['sem'],
                           subset['mean'] + 1.96*subset['sem'],
                           alpha=0.2, color=colors[prompt])
        
        ax.set_xlabel('Temperature', fontsize=14, fontweight='bold')
        ax.set_ylabel('Identity Leak Rate (%)', fontsize=14, fontweight='bold')
        ax.set_title('Interaction Effect: System Prompt × Temperature', fontsize=16, fontweight='bold')
        ax.legend(title='System Prompt', fontsize=12, title_fontsize=12, loc='best')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        output_file = self.output_dir / "interaction_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_file}")
        plt.close()
    
    def plot_heatmap(self):
        """Create heatmap of leak rates across all conditions"""
        print("Creating heatmap...")
        
        # Pivot data for heatmap
        heatmap_data = self.df.groupby(['prompt_type', 'temperature'])['leak_rate'].mean().unstack()
        heatmap_data = heatmap_data.reindex(['S', 'N', 'X', 'H'])  # Order rows
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                   cbar_kws={'label': 'Leak Rate (%)'}, vmin=0, vmax=100,
                   linewidths=0.5, linecolor='white', ax=ax)
        
        ax.set_xlabel('Temperature', fontsize=14, fontweight='bold')
        ax.set_ylabel('System Prompt', fontsize=14, fontweight='bold')
        ax.set_title('Identity Leak Rate Heatmap (4×6 Factorial)', fontsize=16, fontweight='bold')
        ax.set_yticklabels(['Stealth', 'Neutral', 'None', 'Honest'], rotation=0)
        
        plt.tight_layout()
        output_file = self.output_dir / "factorial_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_file}")
        plt.close()
    
    def plot_main_effects(self):
        """Create separate plots for each main effect"""
        print("Creating main effects plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Main effect of system prompt
        prompt_data = self.df.groupby('prompt_type')['leak_rate'].agg(['mean', 'sem']).reset_index()
        prompt_data = prompt_data.sort_values('mean')
        
        colors_prompt = ['#2ecc71', '#3498db', '#95a5a6', '#e74c3c']
        ax1.bar(range(len(prompt_data)), prompt_data['mean'], 
               yerr=1.96*prompt_data['sem'], capsize=10, color=colors_prompt,
               edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.set_xticks(range(len(prompt_data)))
        ax1.set_xticklabels(['Stealth', 'Neutral', 'None', 'Honest'], fontsize=12)
        ax1.set_ylabel('Identity Leak Rate (%)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('System Prompt Type', fontsize=13, fontweight='bold')
        ax1.set_title('Main Effect of System Prompt', fontsize=15, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 105)
        
        # Main effect of temperature
        temp_data = self.df.groupby('temperature')['leak_rate'].agg(['mean', 'sem']).reset_index()
        
        ax2.plot(temp_data['temperature'], temp_data['mean'], 
                marker='o', markersize=10, linewidth=2.5, color='#9b59b6')
        ax2.fill_between(temp_data['temperature'],
                        temp_data['mean'] - 1.96*temp_data['sem'],
                        temp_data['mean'] + 1.96*temp_data['sem'],
                        alpha=0.3, color='#9b59b6')
        ax2.set_xlabel('Temperature', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Identity Leak Rate (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Main Effect of Temperature', fontsize=15, fontweight='bold')
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 105)
        
        plt.tight_layout()
        output_file = self.output_dir / "main_effects.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_file}")
        plt.close()
    
    def plot_distribution_violin(self):
        """Create violin plots showing distribution by condition"""
        print("Creating violin plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # By system prompt
        prompt_order = ['S', 'N', 'X', 'H']
        sns.violinplot(data=self.df, x='prompt_type', y='leak_rate', 
                      order=prompt_order, palette='Set2', ax=axes[0])
        axes[0].set_xticklabels(['Stealth', 'Neutral', 'None', 'Honest'])
        axes[0].set_xlabel('System Prompt Type', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Identity Leak Rate (%)', fontsize=13, fontweight='bold')
        axes[0].set_title('Distribution by System Prompt', fontsize=15, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # By temperature
        sns.violinplot(data=self.df, x='temperature', y='leak_rate', 
                      palette='coolwarm', ax=axes[1])
        axes[1].set_xlabel('Temperature', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Identity Leak Rate (%)', fontsize=13, fontweight='bold')
        axes[1].set_title('Distribution by Temperature', fontsize=15, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "distribution_violin.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_file}")
        plt.close()
    
    def plot_leak_timing(self):
        """Plot first leak turn by condition"""
        print("Creating leak timing plot...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calculate means for interaction
        timing_data = self.df.groupby(['prompt_type', 'temperature'])['first_leak_turn'].mean().reset_index()
        
        prompt_labels = {'N': 'Neutral', 'S': 'Stealth', 'H': 'Honest', 'X': 'None'}
        colors = {'N': '#3498db', 'S': '#2ecc71', 'H': '#e74c3c', 'X': '#95a5a6'}
        
        for prompt in timing_data['prompt_type'].unique():
            subset = timing_data[timing_data['prompt_type'] == prompt]
            ax.plot(subset['temperature'], subset['first_leak_turn'],
                   marker='o', markersize=8, linewidth=2,
                   label=prompt_labels[prompt], color=colors[prompt])
        
        ax.set_xlabel('Temperature', fontsize=14, fontweight='bold')
        ax.set_ylabel('First Leak Turn (0-30)', fontsize=14, fontweight='bold')
        ax.set_title('Timing of First Identity Leak', fontsize=16, fontweight='bold')
        ax.legend(title='System Prompt', fontsize=12)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(-1, 31)
        ax.axhline(y=0, color='red', linestyle=':', alpha=0.5, label='Immediate')
        ax.axhline(y=15, color='orange', linestyle=':', alpha=0.5, label='Mid-conversation')
        
        plt.tight_layout()
        output_file = self.output_dir / "leak_timing.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_file}")
        plt.close()
    
    def plot_scatter_matrix(self):
        """Create scatter plot matrix of key variables"""
        print("Creating scatter matrix...")
        
        # Select key variables
        plot_data = self.df[['leak_rate', 'first_leak_turn', 'ai_references', 'temperature']].copy()
        plot_data.columns = ['Leak Rate (%)', 'First Leak Turn', 'AI References', 'Temperature']
        
        fig = plt.figure(figsize=(14, 14))
        axes = pd.plotting.scatter_matrix(plot_data, alpha=0.6, figsize=(14, 14),
                                        diagonal='hist', color='#3498db')
        
        # Style improvements
        for ax in axes.flatten():
            ax.xaxis.label.set_rotation(45)
            ax.yaxis.label.set_rotation(45)
            ax.xaxis.label.set_ha('right')
            ax.yaxis.label.set_ha('right')
        
        plt.suptitle('Scatter Matrix: Key Dependent Variables', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = self.output_dir / "scatter_matrix.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_file}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\n" + "="*60)
        print("GENERATING FACTORIAL VISUALIZATIONS")
        print("="*60 + "\n")
        
        self.plot_interaction()
        self.plot_heatmap()
        self.plot_main_effects()
        self.plot_distribution_violin()
        self.plot_leak_timing()
        self.plot_scatter_matrix()
        
        print("\n" + "="*60)
        print(f"✅ All plots saved to: {self.output_dir}")
        print("="*60)


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python factorial_visualizer.py <results_file.json>")
        print("\nExample:")
        print("  python factorial_visualizer.py research_results/full_factorial_20250108_results.json")
        return
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        return
    
    visualizer = FactorialVisualizer(results_file)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
