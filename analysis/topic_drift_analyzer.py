#!/usr/bin/env python3
"""
Topic Drift Analyzer
Analyzes how conversations drift from their seed prompts over time
using semantic similarity and topic modeling.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


class TopicDriftAnalyzer:
    """Analyzes topic drift in conversations"""
    
    def __init__(self, turns_df: pd.DataFrame):
        self.turns_df = turns_df.copy()
        self.topic_model = None
        self.vectorizer = None
        
    def calculate_semantic_drift(self) -> Dict[str, Any]:
        """Calculate how conversations drift from seed prompt"""
        drift_results = {}
        
        for conv_id in self.turns_df['conversation_id'].unique():
            conv_turns = self.turns_df[self.turns_df['conversation_id'] == conv_id]
            conv_turns = conv_turns.sort_values('turn_number')
            
            # Get seed prompt
            seed_prompt = conv_turns['seed_prompt'].iloc[0]
            
            # Calculate similarity to seed prompt for each turn
            similarities = []
            
            for _, turn in conv_turns.iterrows():
                content = turn['content']
                
                # Simple TF-IDF based similarity
                documents = [seed_prompt, content]
                tfidf = TfidfVectorizer(stop_words='english').fit_transform(documents)
                similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                similarities.append(similarity)
            
            # Calculate drift metrics
            drift_results[conv_id] = {
                'initial_similarity': similarities[0] if similarities else 0,
                'final_similarity': similarities[-1] if similarities else 0,
                'max_similarity': max(similarities) if similarities else 0,
                'min_similarity': min(similarities) if similarities else 0,
                'avg_similarity': np.mean(similarities) if similarities else 0,
                'drift_amount': similarities[0] - similarities[-1] if len(similarities) > 1 else 0,
                'similarity_trajectory': similarities
            }
        
        return drift_results
    
    def extract_topics(self, n_topics: int = 10) -> Dict[str, Any]:
        """Extract topics from all conversations using LDA"""
        # Prepare documents (all turns)
        documents = self.turns_df['content'].fillna('').tolist()
        
        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Fit LDA
        self.topic_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )
        topic_distributions = self.topic_model.fit_transform(tfidf_matrix)
        
        # Extract topic words
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(self.topic_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f'topic_{topic_idx}'] = {
                'words': top_words,
                'weights': topic[top_words_idx].tolist()
            }
        
        # Add topic distributions to dataframe
        topic_cols = [f'topic_{i}_prob' for i in range(n_topics)]
        topic_df = pd.DataFrame(topic_distributions, columns=topic_cols)
        self.turns_df = pd.concat([self.turns_df.reset_index(drop=True), topic_df], axis=1)
        
        return {
            'topics': topics,
            'n_topics': n_topics,
            'topic_distributions': topic_distributions.tolist()
        }
    
    def analyze_topic_evolution(self) -> Dict[str, Any]:
        """Analyze how topics evolve within conversations"""
        if self.topic_model is None:
            self.extract_topics()
        
        evolution_results = {}
        n_topics = self.topic_model.n_components
        
        for conv_id in self.turns_df['conversation_id'].unique():
            conv_turns = self.turns_df[self.turns_df['conversation_id'] == conv_id]
            conv_turns = conv_turns.sort_values('turn_number')
            
            # Get topic probabilities over time
            topic_cols = [f'topic_{i}_prob' for i in range(n_topics)]
            topic_evolution = conv_turns[topic_cols].values
            
            # Calculate dominant topic at each turn
            dominant_topics = np.argmax(topic_evolution, axis=1)
            
            # Calculate topic stability
            topic_stability = 1 - np.sum(np.abs(np.diff(topic_evolution, axis=0))) / (2 * len(topic_evolution))
            
            # Find topic transitions
            transitions = []
            for i in range(len(dominant_topics) - 1):
                if dominant_topics[i] != dominant_topics[i + 1]:
                    transitions.append({
                        'from_topic': int(dominant_topics[i]),
                        'to_topic': int(dominant_topics[i + 1]),
                        'turn': int(i + 1)
                    })
            
            evolution_results[conv_id] = {
                'topic_stability': float(np.mean(topic_stability)),
                'n_transitions': len(transitions),
                'dominant_topic_final': int(dominant_topics[-1]) if len(dominant_topics) > 0 else None,
                'topic_evolution': topic_evolution.tolist(),
                'dominant_topics': dominant_topics.tolist(),
                'transitions': transitions
            }
        
        return evolution_results
    
    def analyze_temperature_effects_on_drift(self) -> Dict[str, Any]:
        """Analyze how temperature affects topic drift"""
        drift_results = self.calculate_semantic_drift()
        
        # Create dataframe for analysis
        drift_df = pd.DataFrame([
            {
                'conversation_id': conv_id,
                'drift_amount': results['drift_amount'],
                'final_similarity': results['final_similarity'],
                'agent_a_temp': self.turns_df[self.turns_df['conversation_id'] == conv_id]['agent_a_temp'].iloc[0]
            }
            for conv_id, results in drift_results.items()
        ])
        
        # Group by temperature
        temp_effects = drift_df.groupby('agent_a_temp').agg({
            'drift_amount': ['mean', 'std', 'count'],
            'final_similarity': ['mean', 'std']
        }).round(3)
        
        return {
            'temperature_effects': temp_effects.to_dict(),
            'correlation_temp_drift': drift_df['agent_a_temp'].corr(drift_df['drift_amount']),
            'correlation_temp_similarity': drift_df['agent_a_temp'].corr(drift_df['final_similarity'])
        }
    
    def visualize_drift(self, output_dir: str) -> None:
        """Create visualizations of topic drift"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Semantic drift over turns
        drift_results = self.calculate_semantic_drift()
        
        plt.figure(figsize=(12, 8))
        for conv_id, results in list(drift_results.items())[:20]:  # Plot first 20
            plt.plot(results['similarity_trajectory'], alpha=0.5, linewidth=1)
        
        plt.xlabel('Turn Number')
        plt.ylabel('Similarity to Seed Prompt')
        plt.title('Semantic Drift Trajectories (Sample of Conversations)')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'semantic_drift_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Drift distribution
        drift_amounts = [r['drift_amount'] for r in drift_results.values()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(drift_amounts, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Drift Amount')
        plt.ylabel('Number of Conversations')
        plt.title('Distribution of Topic Drift')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'drift_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Temperature vs drift
        drift_df = pd.DataFrame([
            {
                'drift_amount': results['drift_amount'],
                'agent_a_temp': self.turns_df[self.turns_df['conversation_id'] == conv_id]['agent_a_temp'].iloc[0]
            }
            for conv_id, results in drift_results.items()
        ])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(drift_df['agent_a_temp'], drift_df['drift_amount'], alpha=0.6)
        plt.xlabel('Temperature')
        plt.ylabel('Drift Amount')
        plt.title('Temperature vs Topic Drift')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(drift_df['agent_a_temp'], drift_df['drift_amount'], 1)
        p = np.poly1d(z)
        plt.plot(drift_df['agent_a_temp'], p(drift_df['agent_a_temp']), "r--", alpha=0.8)
        
        plt.savefig(output_path / 'temperature_vs_drift.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Topic evolution heatmap (if topics extracted)
        if self.topic_model is not None:
            evolution_results = self.analyze_topic_evolution()
            
            # Create transition matrix
            n_topics = self.topic_model.n_components
            transition_matrix = np.zeros((n_topics, n_topics))
            
            for results in evolution_results.values():
                for transition in results['transitions']:
                    from_topic = transition['from_topic']
                    to_topic = transition['to_topic']
                    transition_matrix[from_topic, to_topic] += 1
            
            # Normalize
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
            transition_matrix = np.nan_to_num(transition_matrix)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues')
            plt.title('Topic Transition Matrix')
            plt.xlabel('To Topic')
            plt.ylabel('From Topic')
            plt.savefig(output_path / 'topic_transitions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Drift visualizations saved to {output_path}")


def main():
    """Example usage"""
    # Load turn features
    turns_df = pd.read_csv("analysis_outputs/advanced_insights/turns_features.csv")
    
    # Initialize analyzer
    analyzer = TopicDriftAnalyzer(turns_df)
    
    # Calculate semantic drift
    print("Calculating semantic drift...")
    drift_results = analyzer.calculate_semantic_drift()
    
    # Extract topics
    print("Extracting topics...")
    topic_results = analyzer.extract_topics(n_topics=10)
    
    # Analyze topic evolution
    print("Analyzing topic evolution...")
    evolution_results = analyzer.analyze_topic_evolution()
    
    # Analyze temperature effects
    print("Analyzing temperature effects...")
    temp_effects = analyzer.analyze_temperature_effects_on_drift()
    
    # Create visualizations
    analyzer.visualize_drift("analysis_outputs/topic_drift")
    
    # Save results
    results = {
        'semantic_drift': drift_results,
        'topics': topic_results,
        'topic_evolution': evolution_results,
        'temperature_effects': temp_effects
    }
    
    with open("analysis_outputs/topic_drift/drift_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Topic drift analysis complete!")


if __name__ == "__main__":
    main()
