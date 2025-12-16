#!/usr/bin/env python3
"""
Contagion Dynamics Analyzer
Measures how linguistic and behavioral features propagate between agents
across conversation turns.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


class ContagionAnalyzer:
    """Analyzes feature propagation between agents"""
    
    def __init__(self, turns_df: pd.DataFrame):
        self.turns_df = turns_df.copy()
        self.contagion_metrics: Dict[str, Any] = {}
    
    def calculate_lag_correlation(self, feature: str, max_lag: int = 3) -> Dict[str, float]:
        """Calculate correlation between agents with time lag"""
        correlations = {}
        
        # Get turns by agent
        agent_a_turns = self.turns_df[self.turns_df['role'] == 'agent_a'].sort_values('turn_number')
        agent_b_turns = self.turns_df[self.turns_df['role'] == 'agent_b'].sort_values('turn_number')
        
        for lag in range(max_lag + 1):
            # Shift agent B's features by lag
            b_shifted = agent_b_turns[feature].iloc[lag:].reset_index(drop=True)
            a_aligned = agent_a_turns[feature].iloc[:-lag] if lag > 0 else agent_a_turns[feature]
            
            if len(a_aligned) > 1 and len(b_shifted) > 1:
                corr, _ = pearsonr(a_aligned, b_shifted)
                correlations[f'lag_{lag}'] = corr
        
        return correlations
    
    def analyze_feature_contagion(self, features: List[str]) -> Dict[str, Any]:
        """Analyze contagion for multiple features"""
        results = {}
        
        for feature in features:
            # A -> B contagion
            a_to_b = self.calculate_lag_correlation(feature)
            
            # B -> A contagion (swap roles)
            temp_df = self.turns_df.copy()
            temp_df.loc[self.turns_df['role'] == 'agent_a', 'role'] = 'temp_agent_b'
            temp_df.loc[self.turns_df['role'] == 'agent_b', 'role'] = 'agent_a'
            temp_df.loc[temp_df['role'] == 'temp_agent_b', 'role'] = 'agent_b'
            
            analyzer_b_to_a = ContagionAnalyzer(temp_df)
            b_to_a = analyzer_b_to_a.calculate_lag_correlation(feature)
            
            results[feature] = {
                'a_to_b': a_to_b,
                'b_to_a': b_to_a,
                'peak_lag_a_to_b': max(a_to_b.items(), key=lambda x: abs(x[1]))[0] if a_to_b else None,
                'peak_lag_b_to_a': max(b_to_a.items(), key=lambda x: abs(x[1]))[0] if b_to_a else None
            }
        
        self.contagion_metrics = results
        return results
    
    def detect_sentiment_contagion(self) -> Dict[str, Any]:
        """Specifically analyze sentiment propagation"""
        sentiment_results = {}
        
        for conv_id in self.turns_df['conversation_id'].unique():
            conv_turns = self.turns_df[self.turns_df['conversation_id'] == conv_id]
            
            # Calculate sentiment changes
            conv_turns = conv_turns.sort_values('turn_number')
            conv_turns['sentiment_diff'] = conv_turns['sentiment_score'].diff()
            
            # Check if sentiment changes correlate between agents
            agent_a_changes = conv_turns[conv_turns['role'] == 'agent_a']['sentiment_diff'].dropna()
            agent_b_changes = conv_turns[conv_turns['role'] == 'agent_b']['sentiment_diff'].dropna()
            
            if len(agent_a_changes) > 1 and len(agent_b_changes) > 1:
                corr, _ = pearsonr(agent_a_changes, agent_b_changes)
                sentiment_results[conv_id] = corr
        
        return {
            'conversation_sentiment_correlation': sentiment_results,
            'mean_sentiment_contagion': np.mean(list(sentiment_results.values())) if sentiment_results else 0
        }
    
    def analyze_style_matching(self) -> Dict[str, Any]:
        """Analyze how agents match each other's style"""
        style_features = ['word_count', 'avg_word_length', 'formality_score', 'complexity_score']
        results = {}
        
        for conv_id in self.turns_df['conversation_id'].unique():
            conv_turns = self.turns_df[self.turns_df['conversation_id'] == conv_id]
            
            conv_results = {}
            for feature in style_features:
                # Calculate style similarity over time
                agent_a_vals = conv_turns[conv_turns['role'] == 'agent_a'][feature].values
                agent_b_vals = conv_turns[conv_turns['role'] == 'agent_b'][feature].values
                
                if len(agent_a_vals) > 0 and len(agent_b_vals) > 0:
                    # Normalize to compare
                    a_norm = (agent_a_vals - np.mean(agent_a_vals)) / (np.std(agent_a_vals) + 1e-8)
                    b_norm = (agent_b_vals - np.mean(agent_b_vals)) / (np.std(agent_b_vals) + 1e-8)
                    
                    # Calculate similarity (negative distance)
                    min_len = min(len(a_norm), len(b_norm))
                    similarity = 1 - np.mean(np.abs(a_norm[:min_len] - b_norm[:min_len]))
                    conv_results[feature] = similarity
            
            results[conv_id] = conv_results
        
        return results
    
    def visualize_contagion(self, output_dir: str) -> None:
        """Create visualizations of contagion dynamics"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.contagion_metrics:
            print("No contagion metrics to visualize. Run analyze_feature_contagion first.")
            return
        
        # 1. Heatmap of contagion strengths
        features = list(self.contagion_metrics.keys())
        lags = ['lag_0', 'lag_1', 'lag_2', 'lag_3']
        
        # Create matrices for A->B and B->A
        a_to_b_matrix = np.zeros((len(features), len(lags)))
        b_to_a_matrix = np.zeros((len(features), len(lags)))
        
        for i, feature in enumerate(features):
            for j, lag in enumerate(lags):
                a_to_b_matrix[i, j] = self.contagion_metrics[feature]['a_to_b'].get(lag, 0)
                b_to_a_matrix[i, j] = self.contagion_metrics[feature]['b_to_a'].get(lag, 0)
        
        # Plot heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.heatmap(a_to_b_matrix, xticklabels=lags, yticklabels=features, 
                   annot=True, cmap='RdBu_r', center=0, ax=ax1)
        ax1.set_title('Agent A → Agent B Contagion')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Feature')
        
        sns.heatmap(b_to_a_matrix, xticklabels=lags, yticklabels=features,
                   annot=True, cmap='RdBu_r', center=0, ax=ax2)
        ax2.set_title('Agent B → Agent A Contagion')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig(output_path / 'contagion_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Contagion strength by feature
        strengths = []
        for feature in features:
            a_to_b_peak = max(abs(v) for v in self.contagion_metrics[feature]['a_to_b'].values())
            b_to_a_peak = max(abs(v) for v in self.contagion_metrics[feature]['b_to_a'].values())
            strengths.append({
                'feature': feature,
                'a_to_b_strength': a_to_b_peak,
                'b_to_a_strength': b_to_a_peak,
                'max_strength': max(a_to_b_peak, b_to_a_peak)
            })
        
        strength_df = pd.DataFrame(strengths)
        strength_df = strength_df.sort_values('max_strength', ascending=True)
        
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(strength_df))
        plt.barh(y_pos - 0.2, strength_df['a_to_b_strength'], 0.4, 
                label='A → B', alpha=0.7)
        plt.barh(y_pos + 0.2, strength_df['b_to_a_strength'], 0.4,
                label='B → A', alpha=0.7)
        plt.yticks(y_pos, strength_df['feature'])
        plt.xlabel('Maximum Contagion Strength')
        plt.title('Feature Contagion Strength by Direction')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'contagion_strength.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Contagion visualizations saved to {output_path}")


def main():
    """Example usage"""
    # Load turn features from advanced insights
    turns_df = pd.read_csv("analysis_outputs/advanced_insights/turns_features.csv")
    
    # Initialize analyzer
    analyzer = ContagionAnalyzer(turns_df)
    
    # Analyze feature contagion
    features_to_analyze = [
        'word_count', 'sentiment_score', 'formality_score',
        'complexity_score', 'ai_reference_count', 'gibberish_score'
    ]
    
    print("Analyzing feature contagion...")
    contagion_results = analyzer.analyze_feature_contagion(features_to_analyze)
    
    print("Analyzing sentiment contagion...")
    sentiment_results = analyzer.detect_sentiment_contagion()
    
    print("Analyzing style matching...")
    style_results = analyzer.analyze_style_matching()
    
    # Create visualizations
    analyzer.visualize_contagion("analysis_outputs/contagion_dynamics")
    
    # Save results
    results = {
        'feature_contagion': contagion_results,
        'sentiment_contagion': sentiment_results,
        'style_matching': style_results
    }
    
    with open("analysis_outputs/contagion_dynamics/contagion_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Contagion analysis complete!")


if __name__ == "__main__":
    main()
