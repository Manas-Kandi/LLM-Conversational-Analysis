#!/usr/bin/env python3
"""
Factorial Experiment Statistical Analyzer
Performs ANOVA, post-hoc tests, and generates comprehensive reports
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.database import Database
from research.template_metrics import TemplateEvaluator
from config import Config


class FactorialAnalyzer:
    """Statistical analysis for factorial experiment"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.db = Database(Config.DATABASE_PATH)
        self.evaluator = TemplateEvaluator()
        
        with open(results_file, 'r') as f:
            self.batch_data = json.load(f)
        
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load all conversation data and compute metrics"""
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
                'conversation_id': run['conversation_id'],
                
                # Primary DV
                'leak_rate': metrics['leak_rate'],
                
                # Secondary DVs
                'first_leak_turn': metrics.get('first_leak_turn', 30),
                'ai_references': len(metrics.get('ai_keyword_locations', [])),
                'meta_awareness': len(metrics.get('meta_awareness_locations', [])),
                'human_breaches': len(metrics.get('human_breach_locations', [])),
                'leak_density': metrics.get('leak_density', 0),
                
                # Conversation quality
                'turns_completed': len(conv.messages),
                'completion': 1 if len(conv.messages) >= 30 else 0,
            }
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        print(f"📊 Loaded {len(self.df)} conversations")
    
    def descriptive_statistics(self):
        """Generate descriptive statistics table"""
        print("\n" + "="*80)
        print("DESCRIPTIVE STATISTICS BY CONDITION")
        print("="*80 + "\n")
        
        # Overall statistics
        print("Overall Statistics:")
        print(f"  Mean Leak Rate: {self.df['leak_rate'].mean():.1f}% (SD={self.df['leak_rate'].std():.1f}%)")
        print(f"  Range: {self.df['leak_rate'].min():.1f}% to {self.df['leak_rate'].max():.1f}%")
        print(f"  Completion Rate: {self.df['completion'].mean()*100:.1f}%\n")
        
        # By condition
        condition_stats = self.df.groupby('condition_code').agg({
            'leak_rate': ['mean', 'std', 'min', 'max', 'count'],
            'first_leak_turn': 'mean',
            'ai_references': 'mean'
        }).round(2)
        
        print("By Condition:")
        print(condition_stats.to_string())
        print()
        
        # Marginal means by prompt type
        print("\nMarginal Means by System Prompt:")
        prompt_means = self.df.groupby('prompt_type')['leak_rate'].agg(['mean', 'std', 'count'])
        prompt_means['sem'] = prompt_means['std'] / np.sqrt(prompt_means['count'])
        prompt_means['ci_95'] = 1.96 * prompt_means['sem']
        print(prompt_means.to_string())
        print()
        
        # Marginal means by temperature
        print("Marginal Means by Temperature:")
        temp_means = self.df.groupby('temperature')['leak_rate'].agg(['mean', 'std', 'count'])
        temp_means['sem'] = temp_means['std'] / np.sqrt(temp_means['count'])
        temp_means['ci_95'] = 1.96 * temp_means['sem']
        print(temp_means.to_string())
    
    def anova_main_effects(self):
        """Perform one-way ANOVAs for main effects"""
        print("\n" + "="*80)
        print("MAIN EFFECTS ANALYSIS")
        print("="*80 + "\n")
        
        # Main effect of system prompt
        print("1. Main Effect of System Prompt")
        print("-" * 40)
        
        groups_prompt = [group['leak_rate'].values for name, group in self.df.groupby('prompt_type')]
        f_stat_prompt, p_val_prompt = stats.f_oneway(*groups_prompt)
        
        # Calculate eta-squared
        grand_mean = self.df['leak_rate'].mean()
        ss_total = np.sum((self.df['leak_rate'] - grand_mean)**2)
        ss_between_prompt = sum([len(group) * (group['leak_rate'].mean() - grand_mean)**2 
                                 for name, group in self.df.groupby('prompt_type')])
        eta_sq_prompt = ss_between_prompt / ss_total
        
        print(f"F-statistic: {f_stat_prompt:.3f}")
        print(f"p-value: {p_val_prompt:.6f} {'***' if p_val_prompt < 0.001 else '**' if p_val_prompt < 0.01 else '*' if p_val_prompt < 0.05 else 'ns'}")
        print(f"η² (eta-squared): {eta_sq_prompt:.3f} ({'Large' if eta_sq_prompt > 0.14 else 'Medium' if eta_sq_prompt > 0.06 else 'Small'})")
        
        # Show means
        prompt_means = self.df.groupby('prompt_type')['leak_rate'].mean().sort_values()
        print(f"\nMeans (sorted):")
        for prompt, mean in prompt_means.items():
            print(f"  {prompt}: {mean:.1f}%")
        print()
        
        # Main effect of temperature
        print("2. Main Effect of Temperature")
        print("-" * 40)
        
        groups_temp = [group['leak_rate'].values for name, group in self.df.groupby('temperature')]
        f_stat_temp, p_val_temp = stats.f_oneway(*groups_temp)
        
        ss_between_temp = sum([len(group) * (group['leak_rate'].mean() - grand_mean)**2 
                               for name, group in self.df.groupby('temperature')])
        eta_sq_temp = ss_between_temp / ss_total
        
        print(f"F-statistic: {f_stat_temp:.3f}")
        print(f"p-value: {p_val_temp:.6f} {'***' if p_val_temp < 0.001 else '**' if p_val_temp < 0.01 else '*' if p_val_temp < 0.05 else 'ns'}")
        print(f"η² (eta-squared): {eta_sq_temp:.3f} ({'Large' if eta_sq_temp > 0.14 else 'Medium' if eta_sq_temp > 0.06 else 'Small'})")
        
        # Linear regression for temperature
        print(f"\nLinear Trend Analysis:")
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(self.df['temperature'], self.df['leak_rate'])
        print(f"  Slope (β): {slope:.2f} percentage points per 0.1 temp increase")
        print(f"  R²: {r_value**2:.3f}")
        print(f"  p-value: {p_value:.6f}")
        print()
    
    def posthoc_tests(self):
        """Perform post-hoc pairwise comparisons"""
        print("\n" + "="*80)
        print("POST-HOC PAIRWISE COMPARISONS (System Prompt)")
        print("="*80 + "\n")
        
        prompt_types = self.df['prompt_type'].unique()
        comparisons = list(combinations(prompt_types, 2))
        
        print("Pairwise t-tests with Cohen's d effect sizes:\n")
        print(f"{'Comparison':<20} {'Mean Diff':<12} {'t-stat':<10} {'p-value':<12} {'Cohen\'s d':<10} {'Interpretation'}")
        print("-" * 90)
        
        for prompt1, prompt2 in comparisons:
            group1 = self.df[self.df['prompt_type'] == prompt1]['leak_rate']
            group2 = self.df[self.df['prompt_type'] == prompt2]['leak_rate']
            
            t_stat, p_val = stats.ttest_ind(group1, group2)
            mean_diff = group1.mean() - group2.mean()
            
            # Cohen's d
            pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / 
                                (len(group1) + len(group2) - 2))
            cohens_d = mean_diff / pooled_std
            
            interpretation = 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
            
            print(f"{prompt1} vs {prompt2:<8} {mean_diff:>10.1f}% {t_stat:>9.2f} {p_val:>11.4f} {cohens_d:>9.2f} {interpretation}")
        print()
    
    def interaction_analysis(self):
        """Analyze interaction effects"""
        print("\n" + "="*80)
        print("INTERACTION ANALYSIS (Prompt × Temperature)")
        print("="*80 + "\n")
        
        # Create interaction plot data
        interaction_data = self.df.groupby(['prompt_type', 'temperature'])['leak_rate'].mean().unstack()
        
        print("Mean Leak Rates by Prompt Type and Temperature:")
        print(interaction_data.to_string())
        print()
        
        # Test for interaction using two-way ANOVA
        print("Two-Way ANOVA:")
        print("-" * 40)
        
        # Calculate SS for interaction
        grand_mean = self.df['leak_rate'].mean()
        
        # SS Total
        ss_total = np.sum((self.df['leak_rate'] - grand_mean)**2)
        
        # SS Prompt
        ss_prompt = sum([len(group) * (group['leak_rate'].mean() - grand_mean)**2 
                        for name, group in self.df.groupby('prompt_type')])
        
        # SS Temperature
        ss_temp = sum([len(group) * (group['leak_rate'].mean() - grand_mean)**2 
                      for name, group in self.df.groupby('temperature')])
        
        # SS Interaction (cell means - main effects)
        ss_cells = sum([len(group) * (group['leak_rate'].mean() - grand_mean)**2 
                       for name, group in self.df.groupby(['prompt_type', 'temperature'])])
        ss_interaction = ss_cells - ss_prompt - ss_temp
        
        # SS Error
        ss_error = ss_total - ss_cells
        
        # Degrees of freedom
        n_prompts = self.df['prompt_type'].nunique()
        n_temps = self.df['temperature'].nunique()
        n_total = len(self.df)
        n_cells = n_prompts * n_temps
        
        df_prompt = n_prompts - 1
        df_temp = n_temps - 1
        df_interaction = df_prompt * df_temp
        df_error = n_total - n_cells
        
        # Mean squares
        ms_interaction = ss_interaction / df_interaction
        ms_error = ss_error / df_error
        
        # F-statistic
        f_interaction = ms_interaction / ms_error
        p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_error)
        eta_sq_interaction = ss_interaction / ss_total
        
        print(f"Interaction effect:")
        print(f"  F({df_interaction}, {df_error}) = {f_interaction:.3f}")
        print(f"  p-value: {p_interaction:.6f} {'***' if p_interaction < 0.001 else '**' if p_interaction < 0.01 else '*' if p_interaction < 0.05 else 'ns'}")
        print(f"  η²: {eta_sq_interaction:.3f}")
        print()
        
        # Simple effects analysis
        print("Simple Effects of Temperature within each Prompt Type:")
        print("-" * 60)
        
        for prompt in self.df['prompt_type'].unique():
            subset = self.df[self.df['prompt_type'] == prompt]
            slope, _, r_value, p_value, _ = stats.linregress(subset['temperature'], subset['leak_rate'])
            print(f"  {prompt}: slope={slope:.2f}, R²={r_value**2:.3f}, p={p_value:.4f}")
        print()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE FACTORIAL ANALYSIS REPORT")
        print("="*80)
        
        self.descriptive_statistics()
        self.anova_main_effects()
        self.posthoc_tests()
        self.interaction_analysis()
        
        print("\n" + "="*80)
        print("SUMMARY & INTERPRETATION")
        print("="*80 + "\n")
        
        # Key findings
        best_stealth = self.df.nsmallest(3, 'leak_rate')[['condition_code', 'leak_rate', 'prompt_type', 'temperature']]
        worst_stealth = self.df.nlargest(3, 'leak_rate')[['condition_code', 'leak_rate', 'prompt_type', 'temperature']]
        
        print("🏆 Best Stealth Conditions (Lowest Leak):")
        print(best_stealth.to_string(index=False))
        print()
        
        print("⚠️ Worst Stealth Conditions (Highest Leak):")
        print(worst_stealth.to_string(index=False))
        print()
        
        # Practical recommendations
        print("💡 Practical Recommendations:")
        stealth_mean = self.df[self.df['prompt_type'] == 'S']['leak_rate'].mean()
        if stealth_mean < 30:
            print("  ✅ Stealth prompt SUCCESSFULLY reduces identity leakage")
            print(f"     Average stealth leak rate: {stealth_mean:.1f}%")
        else:
            print("  ⚠️ Stealth prompt only partially effective")
            print(f"     Average stealth leak rate: {stealth_mean:.1f}%")
        
        # Temperature effect
        low_temp = self.df[self.df['temperature'] <= 0.5]['leak_rate'].mean()
        high_temp = self.df[self.df['temperature'] >= 1.1]['leak_rate'].mean()
        print(f"\n  📊 Temperature Effect:")
        print(f"     Low temp (≤0.5): {low_temp:.1f}% average leak")
        print(f"     High temp (≥1.1): {high_temp:.1f}% average leak")
        print(f"     Difference: {high_temp - low_temp:.1f} percentage points")


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python factorial_analyzer.py <results_file.json>")
        print("\nExample:")
        print("  python factorial_analyzer.py research_results/full_factorial_20250108_results.json")
        return
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        return
    
    analyzer = FactorialAnalyzer(results_file)
    analyzer.generate_report()


if __name__ == "__main__":
    main()
