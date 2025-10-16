"""
Statistical Analysis
Quantitative metrics without LLM calls
"""
from typing import Dict, Any, List
import statistics
from collections import Counter

from analysis.analyzer import BaseAnalyzer
from storage.models import AnalysisResult, AgentRole


class StatisticalAnalyzer(BaseAnalyzer):
    """Performs statistical analysis on conversation metrics"""
    
    @property
    def analysis_type(self) -> str:
        return "statistical"
    
    def analyze(self) -> AnalysisResult:
        """
        Perform statistical analysis on conversation
        
        Returns:
            AnalysisResult with statistical metrics
        """
        # Calculate various metrics
        metrics = {
            "basic_stats": self._calculate_basic_metrics(),
            "turn_metrics": self._calculate_turn_metrics(),
            "agent_comparison": self._compare_agents(),
            "token_analysis": self._analyze_tokens(),
            "vocabulary_analysis": self._analyze_vocabulary(),
            "timing_analysis": self._analyze_timing()
        }
        
        # Generate summary
        summary = self._generate_summary(metrics)
        
        # Save to database
        analysis_id = self.save_results(metrics, summary)
        
        return AnalysisResult(
            id=analysis_id,
            conversation_id=self.conversation.id,
            analysis_type=self.analysis_type,
            results=metrics,
            summary=summary
        )
    
    def _calculate_basic_metrics(self) -> Dict[str, Any]:
        """Basic conversation statistics"""
        messages = self.conversation.messages
        
        if not messages:
            return {}
        
        return {
            "total_turns": len(messages),
            "duration_seconds": self.conversation.get_duration(),
            "agent_a_turns": len([m for m in messages if m.role == AgentRole.AGENT_A]),
            "agent_b_turns": len([m for m in messages if m.role == AgentRole.AGENT_B]),
            "status": self.conversation.status
        }
    
    def _calculate_turn_metrics(self) -> Dict[str, Any]:
        """Message length and structure metrics"""
        messages = self.conversation.messages
        
        # Character counts
        char_counts = [len(m.content) for m in messages]
        
        # Word counts (approximate)
        word_counts = [len(m.content.split()) for m in messages]
        
        # Sentence counts (approximate by periods)
        sentence_counts = [m.content.count('.') + m.content.count('!') + m.content.count('?') for m in messages]
        
        return {
            "avg_chars_per_turn": statistics.mean(char_counts) if char_counts else 0,
            "median_chars_per_turn": statistics.median(char_counts) if char_counts else 0,
            "min_chars": min(char_counts) if char_counts else 0,
            "max_chars": max(char_counts) if char_counts else 0,
            "stdev_chars": statistics.stdev(char_counts) if len(char_counts) > 1 else 0,
            "avg_words_per_turn": statistics.mean(word_counts) if word_counts else 0,
            "avg_sentences_per_turn": statistics.mean(sentence_counts) if sentence_counts else 0,
            "total_characters": sum(char_counts),
            "total_words": sum(word_counts)
        }
    
    def _compare_agents(self) -> Dict[str, Any]:
        """Compare metrics between Agent A and Agent B"""
        agent_a_msgs = [m for m in self.conversation.messages if m.role == AgentRole.AGENT_A]
        agent_b_msgs = [m for m in self.conversation.messages if m.role == AgentRole.AGENT_B]
        
        if not agent_a_msgs or not agent_b_msgs:
            return {}
        
        # Character counts
        a_chars = [len(m.content) for m in agent_a_msgs]
        b_chars = [len(m.content) for m in agent_b_msgs]
        
        # Word counts
        a_words = [len(m.content.split()) for m in agent_a_msgs]
        b_words = [len(m.content.split()) for m in agent_b_msgs]
        
        # Questions (count question marks)
        a_questions = sum(m.content.count('?') for m in agent_a_msgs)
        b_questions = sum(m.content.count('?') for m in agent_b_msgs)
        
        return {
            "agent_a": {
                "turns": len(agent_a_msgs),
                "avg_chars": statistics.mean(a_chars),
                "avg_words": statistics.mean(a_words),
                "total_questions": a_questions,
                "questions_per_turn": a_questions / len(agent_a_msgs)
            },
            "agent_b": {
                "turns": len(agent_b_msgs),
                "avg_chars": statistics.mean(b_chars),
                "avg_words": statistics.mean(b_words),
                "total_questions": b_questions,
                "questions_per_turn": b_questions / len(agent_b_msgs)
            },
            "comparison": {
                "length_ratio_a_to_b": statistics.mean(a_chars) / statistics.mean(b_chars) if b_chars else 0,
                "verbosity_leader": "agent_a" if statistics.mean(a_chars) > statistics.mean(b_chars) else "agent_b",
                "question_ratio_a_to_b": a_questions / b_questions if b_questions > 0 else float('inf'),
                "more_inquisitive": "agent_a" if a_questions > b_questions else "agent_b"
            }
        }
    
    def _analyze_tokens(self) -> Dict[str, Any]:
        """Token usage analysis"""
        messages = self.conversation.messages
        
        token_counts = [m.token_count for m in messages if m.token_count]
        
        if not token_counts:
            return {"note": "No token count data available"}
        
        return {
            "total_tokens": sum(token_counts),
            "avg_tokens_per_turn": statistics.mean(token_counts),
            "median_tokens_per_turn": statistics.median(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts)
        }
    
    def _analyze_vocabulary(self) -> Dict[str, Any]:
        """Vocabulary diversity analysis"""
        # Combine all text
        all_text = " ".join(m.content for m in self.conversation.messages)
        
        # Simple word tokenization (lowercase, remove punctuation)
        words = all_text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '').split()
        
        # Count unique words
        unique_words = set(words)
        word_freq = Counter(words)
        
        # Most common words
        most_common = word_freq.most_common(10)
        
        # Type-token ratio (lexical diversity)
        ttr = len(unique_words) / len(words) if words else 0
        
        return {
            "total_words": len(words),
            "unique_words": len(unique_words),
            "type_token_ratio": ttr,
            "lexical_diversity": "high" if ttr > 0.6 else "medium" if ttr > 0.4 else "low",
            "most_common_words": [{"word": w, "count": c} for w, c in most_common]
        }
    
    def _analyze_timing(self) -> Dict[str, Any]:
        """Timing and pacing analysis"""
        messages = self.conversation.messages
        
        if len(messages) < 2:
            return {}
        
        # Calculate time deltas between turns
        time_deltas = []
        for i in range(1, len(messages)):
            delta = (messages[i].timestamp - messages[i-1].timestamp).total_seconds()
            time_deltas.append(delta)
        
        duration = self.conversation.get_duration()
        
        return {
            "total_duration_seconds": duration,
            "avg_seconds_per_turn": statistics.mean(time_deltas) if time_deltas else 0,
            "median_seconds_per_turn": statistics.median(time_deltas) if time_deltas else 0,
            "min_turn_gap": min(time_deltas) if time_deltas else 0,
            "max_turn_gap": max(time_deltas) if time_deltas else 0,
            "turns_per_minute": (len(messages) / duration * 60) if duration else 0
        }
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        summary_parts = []
        
        basic = metrics.get("basic_stats", {})
        turns = metrics.get("turn_metrics", {})
        comparison = metrics.get("agent_comparison", {})
        vocab = metrics.get("vocabulary_analysis", {})
        
        # Basic stats
        if basic:
            summary_parts.append(
                f"{basic['total_turns']} turns "
                f"(A: {basic.get('agent_a_turns', 0)}, B: {basic.get('agent_b_turns', 0)})"
            )
        
        # Average length
        if turns:
            summary_parts.append(
                f"Avg: {turns.get('avg_words_per_turn', 0):.0f} words/turn"
            )
        
        # Vocabulary
        if vocab:
            summary_parts.append(
                f"Vocabulary: {vocab.get('unique_words', 0)} unique words "
                f"({vocab.get('lexical_diversity', 'unknown')} diversity)"
            )
        
        # Agent comparison
        if comparison and 'comparison' in comparison:
            comp = comparison['comparison']
            summary_parts.append(
                f"Verbosity: {comp['verbosity_leader'].replace('_', ' ').title()}"
            )
            summary_parts.append(
                f"Questions: {comp['more_inquisitive'].replace('_', ' ').title()}"
            )
        
        return " | ".join(summary_parts)
