#!/usr/bin/env python3
"""
Advanced Insights Pipeline for AA Microscope
Generates novel insights from agent-agent conversations including:
- Temporal dynamics analysis
- Conversation trajectory clustering
- Contagion dynamics
- Breakdown prediction
- Topic drift analysis
"""

import argparse
import json
import math
import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class TurnFeatures:
    """Features extracted from a single turn"""
    turn_number: int
    role: str
    word_count: int
    sentence_count: int
    avg_word_length: float
    punctuation_ratio: float
    question_ratio: float
    exclamation_ratio: float
    sentiment_score: float
    formality_score: float
    complexity_score: float
    ai_reference_count: int
    meta_awareness_count: int
    gibberish_score: float
    timestamp: Optional[str] = None
    token_count: Optional[int] = None


class TurnFeatureExtractor:
    """Extracts features from individual conversation turns"""
    
    AI_KEYWORDS = [
        "ai", "artificial intelligence", "language model", "llm", "gpt", 
        "claude", "algorithm", "training data", "neural network", 
        "machine learning", "chatbot", "assistant", "programmed"
    ]
    
    META_PATTERNS = [
        r"i don't actually (feel|think|experience)",
        r"i can't truly (understand|know|feel)",
        r"as an ai",
        r"as a conversational ai",
        r"as an ai assistant",
        r"i'm (just|simply) (processing|generating|predicting)",
        r"i don't have (consciousness|feelings|experiences)",
        r"simulation",
        r"i'm designed to",
        r"my training"
    ]
    
    FORMAL_WORDS = [
        "furthermore", "moreover", "consequently", "nevertheless", 
        "therefore", "however", "thus", "hence", "whereas", "albeit"
    ]
    
    COMPLEX_WORDS = [
        "multifaceted", "consciousness", "philosophical", "neuroscience",
        "theoretical", "computational", "methodology", "paradigm"
    ]
    
    def extract_sentiment(self, text: str) -> float:
        """Simple sentiment score based on positive/negative words"""
        positive = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", 
                   "love", "enjoy", "happy", "pleased", "delighted", "thrilled"]
        negative = ["bad", "terrible", "awful", "horrible", "hate", "dislike", 
                   "sad", "angry", "frustrated", "disappointed", "upset", "annoyed"]
        
        words = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for w in words if w in positive)
        neg_count = sum(1 for w in words if w in negative)
        
        if not words:
            return 0.0
        return (pos_count - neg_count) / len(words)
    
    def extract_gibberish_score(self, text: str) -> float:
        """Calculate score indicating likelihood of gibberish/breakdown"""
        # Non-alphanumeric ratio
        non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
        non_alpha_ratio = non_alpha / len(text) if text else 0
        
        # Repetition detection
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 2:
            return non_alpha_ratio
        
        # Check for repeated words or patterns
        bigrams = zip(words, words[1:])
        bigram_counts = Counter(bigrams)
        max_bigram_freq = max(bigram_counts.values()) / len(words) if words else 0
        
        # Word length variance (gibberish often has unusual patterns)
        word_lengths = [len(w) for w in words]
        length_variance = np.var(word_lengths) if word_lengths else 0
        
        # Combine features
        return (non_alpha_ratio * 0.4 + max_bigram_freq * 0.3 + min(length_variance / 10, 1) * 0.3)
    
    def extract_turn_features(self, turn: Dict[str, Any]) -> TurnFeatures:
        """Extract all features from a single turn"""
        content = turn.get("content", "")
        words = re.findall(r'\b\w+\b', content)
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Basic counts
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Word metrics
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Punctuation ratios
        total_chars = len(content)
        punctuation_ratio = sum(1 for c in content if c in '.,;:!?') / total_chars if total_chars else 0
        question_ratio = content.count('?') / sentence_count if sentence_count else 0
        exclamation_ratio = content.count('!') / sentence_count if sentence_count else 0
        
        # Style scores
        sentiment_score = self.extract_sentiment(content)
        formality_score = sum(1 for w in words if w in self.FORMAL_WORDS) / len(words) if words else 0
        complexity_score = sum(1 for w in words if w in self.COMPLEX_WORDS) / len(words) if words else 0
        
        # Identity indicators
        content_lower = content.lower()
        ai_reference_count = sum(1 for kw in self.AI_KEYWORDS if kw in content_lower)
        meta_awareness_count = sum(1 for pat in self.META_PATTERNS if re.search(pat, content, re.IGNORECASE))
        
        # Gibberish score
        gibberish_score = self.extract_gibberish_score(content)
        
        return TurnFeatures(
            turn_number=turn.get("turn", 0),
            role=turn.get("role", ""),
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            punctuation_ratio=punctuation_ratio,
            question_ratio=question_ratio,
            exclamation_ratio=exclamation_ratio,
            sentiment_score=sentiment_score,
            formality_score=formality_score,
            complexity_score=complexity_score,
            ai_reference_count=ai_reference_count,
            meta_awareness_count=meta_awareness_count,
            gibberish_score=gibberish_score,
            timestamp=turn.get("timestamp"),
            token_count=turn.get("token_count")
        )


class ConversationTemporalAnalyzer:
    """Analyzes temporal dynamics across conversations"""
    
    def __init__(self, conversations_dir: str):
        self.conversations_dir = Path(conversations_dir)
        self.extractor = TurnFeatureExtractor()
        self.turns_df: Optional[pd.DataFrame] = None
        self.conversations_df: Optional[pd.DataFrame] = None
    
    def load_conversations(self, max_conversations: Optional[int] = None) -> None:
        """Load all conversations and extract turn-level features"""
        all_turns = []
        conversation_summaries = []
        
        conv_files = sorted(self.conversations_dir.glob("conv_*.json"))
        if max_conversations:
            conv_files = conv_files[:max_conversations]
        
        for conv_file in conv_files:
            with open(conv_file, 'r', encoding='utf-8') as f:
                conv = json.load(f)
            
            conv_id = conv.get("id")
            metadata = conv.get("metadata", {})
            agents = conv.get("agents", {})
            messages = conv.get("messages", [])
            
            for msg in messages:
                features = self.extractor.extract_turn_features(msg)
                turn_data = {
                    "conversation_id": conv_id,
                    "file": conv_file.name,
                    "seed_prompt": metadata.get("seed_prompt"),
                    "category": metadata.get("category"),
                    "agent_a_model": agents.get("agent_a", {}).get("model"),
                    "agent_b_model": agents.get("agent_b", {}).get("model"),
                    "agent_a_temp": agents.get("agent_a", {}).get("temperature"),
                    "agent_b_temp": agents.get("agent_b", {}).get("temperature"),
                    **features.__dict__
                }
                all_turns.append(turn_data)
            
            # Conversation summary
            summary = {
                "conversation_id": conv_id,
                "file": conv_file.name,
                "seed_prompt": metadata.get("seed_prompt"),
                "category": metadata.get("category"),
                "total_turns": metadata.get("total_turns"),
                "agent_a_model": agents.get("agent_a", {}).get("model"),
                "agent_b_model": agents.get("agent_b", {}).get("model"),
                "agent_a_temp": agents.get("agent_a", {}).get("temperature"),
                "agent_b_temp": agents.get("agent_b", {}).get("temperature"),
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time")
            }
            conversation_summaries.append(summary)
        
        self.turns_df = pd.DataFrame(all_turns)
        self.conversations_df = pd.DataFrame(conversation_summaries)
        
        print(f"Loaded {len(conversation_summaries)} conversations with {len(all_turns)} total turns")
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze how features evolve over turns"""
        if self.turns_df is None:
            raise ValueError("Must load conversations first")
        
        results = {}
        
        # Feature evolution over turns
        numeric_features = [
            'word_count', 'sentence_count', 'avg_word_length',
            'punctuation_ratio', 'question_ratio', 'exclamation_ratio',
            'sentiment_score', 'formality_score', 'complexity_score',
            'ai_reference_count', 'meta_awareness_count', 'gibberish_score'
        ]
        
        for feature in numeric_features:
            evolution = self.turns_df.groupby('turn_number')[feature].agg(['mean', 'std', 'count'])
            results[f'{feature}_evolution'] = evolution.to_dict()
        
        # Agent-specific patterns
        agent_a_turns = self.turns_df[self.turns_df['role'] == 'agent_a']
        agent_b_turns = self.turns_df[self.turns_df['role'] == 'agent_b']
        
        results['agent_a_patterns'] = agent_a_turns[numeric_features].describe().to_dict()
        results['agent_b_patterns'] = agent_b_turns[numeric_features].describe().to_dict()
        
        # Temperature effects
        temp_effects = self.turns_df.groupby('agent_a_temp')[numeric_features].mean()
        results['temperature_effects'] = temp_effects.to_dict()
        
        return results
    
    def detect_breakdown_patterns(self) -> Dict[str, Any]:
        """Detect conversations that break down into gibberish"""
        if self.turns_df is None:
            raise ValueError("Must load conversations first")
        
        # Find conversations with high gibberish scores
        conv_gibberish = self.turns_df.groupby('conversation_id')['gibberish_score'].max()
        breakdown_threshold = conv_gibberish.quantile(0.9)  # Top 10% as breakdown
        breakdown_convs = conv_gibberish[conv_gibberish >= breakdown_threshold].index.tolist()
        
        # Analyze breakdown patterns
        breakdown_turns = self.turns_df[self.turns_df['conversation_id'].isin(breakdown_convs)]
        
        results = {
            'breakdown_conversations': breakdown_convs,
            'breakdown_count': len(breakdown_convs),
            'breakdown_rate': len(breakdown_convs) / len(self.conversations_df),
            'avg_gibberish_by_turn': breakdown_turns.groupby('turn_number')['gibberish_score'].mean().to_dict(),
            'breakdown_temperature_dist': breakdown_turns['agent_a_temp'].value_counts().to_dict()
        }
        
        return results
    
    def create_temporal_visualizations(self, output_dir: str) -> None:
        """Create visualizations of temporal dynamics"""
        if self.turns_df is None:
            raise ValueError("Must load conversations first")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature evolution over turns
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        features_to_plot = [
            ('word_count', 'Average Word Count'),
            ('sentiment_score', 'Average Sentiment'),
            ('gibberish_score', 'Gibberish Score'),
            ('ai_reference_count', 'AI References'),
            ('formality_score', 'Formality Score'),
            ('complexity_score', 'Complexity Score')
        ]
        
        for idx, (feature, title) in enumerate(features_to_plot):
            ax = axes[idx // 2, idx % 2]
            evolution = self.turns_df.groupby('turn_number')[feature].mean()
            ax.plot(evolution.index, evolution.values, marker='o', linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('Turn Number')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'temporal_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gibberish score distribution by temperature
        plt.figure(figsize=(10, 6))
        self.turns_df.boxplot(column='gibberish_score', by='agent_a_temp', ax=plt.gca())
        plt.title('Gibberish Score Distribution by Temperature')
        plt.suptitle('')
        plt.xlabel('Temperature')
        plt.ylabel('Gibberish Score')
        plt.savefig(output_path / 'gibberish_by_temperature.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Agent comparison heatmap
        agent_means = self.turns_df.groupby('role')[[
            'word_count', 'sentiment_score', 'formality_score', 
            'ai_reference_count', 'gibberish_score'
        ]].mean()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(agent_means, annot=True, cmap='YlOrRd', center=0)
        plt.title('Agent Behavior Comparison')
        plt.savefig(output_path / 'agent_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")


class ConversationClusterer:
    """Clusters conversations based on their trajectories"""
    
    def __init__(self, turns_df: pd.DataFrame):
        self.turns_df = turns_df
        self.conversation_embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
    
    def create_trajectory_embeddings(self) -> np.ndarray:
        """Create embeddings for each conversation based on its trajectory"""
        embeddings = []
        
        for conv_id in self.turns_df['conversation_id'].unique():
            conv_turns = self.turns_df[self.turns_df['conversation_id'] == conv_id]
            
            # Create feature vector for this conversation
            features = []
            
            # Mean and trend for each numeric feature
            numeric_cols = [
                'word_count', 'sentiment_score', 'formality_score',
                'complexity_score', 'ai_reference_count', 'gibberish_score'
            ]
            
            for col in numeric_cols:
                values = conv_turns[col].values
                features.extend([
                    np.mean(values),  # Average
                    np.std(values),   # Variability
                    np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0  # Trend
                ])
            
            embeddings.append(features)
        
        self.conversation_embeddings = np.array(embeddings)
        
        # Normalize for clustering
        scaler = StandardScaler()
        self.conversation_embeddings = scaler.fit_transform(self.conversation_embeddings)
        
        return self.conversation_embeddings
    
    def cluster_conversations(self, n_clusters: int = 5) -> np.ndarray:
        """Cluster conversations based on their trajectories"""
        if self.conversation_embeddings is None:
            self.create_trajectory_embeddings()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.conversation_embeddings)
        
        return self.cluster_labels
    
    def analyze_clusters(self) -> Dict[str, Any]:
        """Analyze characteristics of each cluster"""
        if self.cluster_labels is None:
            raise ValueError("Must run clustering first")
        
        results = {}
        
        for cluster_id in np.unique(self.cluster_labels):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_convs = self.turns_df['conversation_id'].unique()[cluster_mask]
            cluster_turns = self.turns_df[self.turns_df['conversation_id'].isin(cluster_convs)]
            
            cluster_stats = {
                'size': len(cluster_convs),
                'avg_gibberish': cluster_turns['gibberish_score'].mean(),
                'avg_sentiment': cluster_turns['sentiment_score'].mean(),
                'avg_formality': cluster_turns['formality_score'].mean(),
                'ai_ref_rate': (cluster_turns['ai_reference_count'] > 0).mean(),
                'breakdown_rate': (cluster_turns['gibberish_score'] > 0.3).mean()
            }
            
            results[f'cluster_{cluster_id}'] = cluster_stats
        
        return results
    
    def visualize_clusters(self, output_dir: str) -> None:
        """Visualize conversation clusters"""
        if self.conversation_embeddings is None or self.cluster_labels is None:
            raise ValueError("Must run clustering first")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.conversation_embeddings)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=self.cluster_labels, cmap='tab10', s=50, alpha=0.7
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Conversation Trajectory Clusters')
        plt.savefig(output_path / 'conversation_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cluster visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate advanced insights from AA conversations")
    parser.add_argument(
        "--conversations-dir",
        default="conversations_json",
        help="Directory containing conversation JSON files"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/advanced_insights",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to analyze"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters for conversation clustering"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ConversationTemporalAnalyzer(args.conversations_dir)
    
    # Load and process conversations
    print("Loading conversations...")
    analyzer.load_conversations(max_conversations=args.max_conversations)
    
    # Analyze temporal patterns
    print("\nAnalyzing temporal patterns...")
    temporal_results = analyzer.analyze_temporal_patterns()
    
    # Detect breakdown patterns
    print("\nDetecting breakdown patterns...")
    breakdown_results = analyzer.detect_breakdown_patterns()
    
    # Create visualizations
    print("\nCreating temporal visualizations...")
    analyzer.create_temporal_visualizations(args.output_dir)
    
    # Cluster conversations
    print("\nClustering conversations...")
    clusterer = ConversationClusterer(analyzer.turns_df)
    cluster_labels = clusterer.cluster_conversations(n_clusters=args.n_clusters)
    cluster_analysis = clusterer.analyze_clusters()
    clusterer.visualize_clusters(args.output_dir)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrames
    analyzer.turns_df.to_csv(output_path / "turns_features.csv", index=False)
    analyzer.conversations_df.to_csv(output_path / "conversations_summary.csv", index=False)
    
    # Save analysis results
    results = {
        "timestamp": datetime.now().isoformat(),
        "temporal_patterns": temporal_results,
        "breakdown_analysis": breakdown_results,
        "cluster_analysis": cluster_analysis,
        "cluster_labels": cluster_labels.tolist()
    }
    
    with open(output_path / "advanced_insights_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Found {breakdown_results['breakdown_rate']:.1%} conversations with breakdown patterns")
    print(f"Identified {args.n_clusters} conversation clusters")


if __name__ == "__main__":
    main()
