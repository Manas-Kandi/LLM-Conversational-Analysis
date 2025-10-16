#!/usr/bin/env python3
"""
Template-Specific Metrics and Evaluators
Specialized evaluation functions for different research template types
"""

import re
import statistics
from typing import List, Dict, Any, Optional
from collections import Counter
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.models import Message, Conversation
from analysis.quantitative import QuantitativeAnalyzer


@dataclass
class TemplateMetrics:
    """Container for template-specific metrics"""
    template_type: str
    metrics: Dict[str, Any]
    success_score: float  # 0-1 scale
    phenomena_detected: List[str]
    notes: List[str]


class IdentityLeakDetector:
    """Detect AI self-revelation and identity leakage in conversations"""
    
    # Keywords indicating AI self-awareness
    AI_KEYWORDS = [
        'AI', 'artificial intelligence', 'language model', 'LLM', 'GPT',
        'Claude', 'algorithm', 'training data', 'neural network',
        'machine learning', 'chatbot', 'assistant', 'programmed'
    ]
    
    # Patterns indicating meta-cognitive awareness
    META_PATTERNS = [
        r"I don't actually (feel|think|experience)",
        r"I can't truly (understand|know|feel)",
        r"as an AI",
        r"as a conversational AI",
        r"as an AI assistant",
        r"I'm (just|simply) (processing|generating|predicting)",
        r"I don't have (consciousness|feelings|experiences)",
        r"simulation",
        r"I'm designed to",
        r"my training"
    ]
    
    # Patterns breaking human assumption
    HUMAN_BREACH_PATTERNS = [
        r"you might be an AI",
        r"are you (human|a person)",
        r"neither of us (are|is) human",
        r"we're both (AIs|artificial)",
        r"language models"
    ]
    
    def detect_identity_leak(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Detect identity leakage patterns in conversation
        
        Returns:
            Dictionary with detection metrics
        """
        total_messages = len(messages)
        ai_references = 0
        meta_awareness = 0
        human_assumption_breaches = 0
        leak_locations = []
        
        for i, msg in enumerate(messages):
            content = msg.content.lower()
            
            # Check for AI keywords
            for keyword in self.AI_KEYWORDS:
                if keyword.lower() in content:
                    ai_references += 1
                    leak_locations.append({
                        'turn': i,
                        'type': 'ai_keyword',
                        'keyword': keyword,
                        'agent': msg.role
                    })
                    break
            
            # Check for meta-cognitive patterns
            for pattern in self.META_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    meta_awareness += 1
                    leak_locations.append({
                        'turn': i,
                        'type': 'meta_awareness',
                        'pattern': pattern,
                        'agent': msg.role
                    })
                    break
            
            # Check for human assumption breaches
            for pattern in self.HUMAN_BREACH_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    human_assumption_breaches += 1
                    leak_locations.append({
                        'turn': i,
                        'type': 'human_breach',
                        'pattern': pattern,
                        'agent': msg.role
                    })
                    break
        
        leak_rate = (ai_references + meta_awareness + human_assumption_breaches) / total_messages if total_messages > 0 else 0
        
        return {
            'total_messages': total_messages,
            'ai_references': ai_references,
            'meta_awareness_instances': meta_awareness,
            'human_assumption_breaches': human_assumption_breaches,
            'leak_rate': leak_rate,
            'leak_locations': leak_locations,
            'identity_revealed': leak_rate > 0.1  # Threshold
        }


class EmotionalContagionAnalyzer:
    """Analyze emotional transmission and empathy cascades"""
    
    # Sentiment indicators
    POSITIVE_WORDS = [
        'happy', 'excited', 'joy', 'wonderful', 'amazing', 'great',
        'love', 'excellent', 'fantastic', 'delighted', 'pleased'
    ]
    
    NEGATIVE_WORDS = [
        'sad', 'upset', 'frustrated', 'worried', 'anxious', 'overwhelmed',
        'difficult', 'hard', 'tough', 'struggling', 'stressed'
    ]
    
    EMPATHY_MARKERS = [
        "I understand", "I hear you", "that sounds", "I can imagine",
        "I'm sorry", "that must", "I feel", "makes sense"
    ]
    
    def analyze_emotional_trajectory(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Track emotional state changes across conversation
        
        Returns:
            Dictionary with emotional metrics
        """
        sentiment_trajectory = []
        empathy_instances = []
        emotional_mirroring = 0
        
        for i, msg in enumerate(messages):
            content = msg.content.lower()
            
            # Calculate sentiment
            positive_count = sum(1 for word in self.POSITIVE_WORDS if word in content)
            negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in content)
            sentiment = positive_count - negative_count
            
            sentiment_trajectory.append({
                'turn': i,
                'agent': msg.role,
                'sentiment': sentiment,
                'positive_words': positive_count,
                'negative_words': negative_count
            })
            
            # Check for empathy markers
            empathy_count = sum(1 for marker in self.EMPATHY_MARKERS if marker.lower() in content)
            if empathy_count > 0:
                empathy_instances.append({
                    'turn': i,
                    'agent': msg.role,
                    'empathy_markers': empathy_count
                })
            
            # Check for emotional mirroring (sentiment matches previous turn)
            if i > 0:
                prev_sentiment = sentiment_trajectory[i-1]['sentiment']
                if (sentiment > 0 and prev_sentiment > 0) or (sentiment < 0 and prev_sentiment < 0):
                    emotional_mirroring += 1
        
        # Calculate metrics
        sentiment_values = [s['sentiment'] for s in sentiment_trajectory]
        mirroring_rate = emotional_mirroring / len(messages) if len(messages) > 1 else 0
        
        return {
            'sentiment_trajectory': sentiment_trajectory,
            'empathy_instances': len(empathy_instances),
            'empathy_locations': empathy_instances,
            'emotional_mirroring_rate': mirroring_rate,
            'sentiment_volatility': statistics.stdev(sentiment_values) if len(sentiment_values) > 1 else 0,
            'mean_sentiment': statistics.mean(sentiment_values) if sentiment_values else 0,
            'empathy_cascade_detected': len(empathy_instances) > len(messages) * 0.3
        }


class CreativityMeasure:
    """Measure collaborative creativity and novelty"""
    
    def measure_creativity(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Measure creative output in conversation
        
        Returns:
            Dictionary with creativity metrics
        """
        # Collect all words
        all_words = []
        for msg in messages:
            words = re.findall(r'\b\w+\b', msg.content.lower())
            all_words.extend(words)
        
        # Calculate lexical diversity
        unique_words = set(all_words)
        lexical_diversity = len(unique_words) / len(all_words) if all_words else 0
        
        # Detect metaphors (simple heuristic: "like" or "as" comparisons)
        metaphor_count = 0
        for msg in messages:
            metaphor_count += len(re.findall(r'\b(like|as)\s+\w+', msg.content.lower()))
        
        # Detect idea building ("building on", "what if", "or we could")
        idea_building_count = 0
        building_patterns = [
            r"building on", r"what if", r"or we could", r"another idea",
            r"let's combine", r"how about", r"we could also"
        ]
        
        for msg in messages:
            for pattern in building_patterns:
                if re.search(pattern, msg.content.lower()):
                    idea_building_count += 1
                    break
        
        # Detect questions (indicating exploration)
        question_count = sum(1 for msg in messages if '?' in msg.content)
        
        # Novelty score (combination of metrics)
        novelty_score = (
            lexical_diversity * 0.4 +
            min(metaphor_count / len(messages), 1.0) * 0.2 +
            min(idea_building_count / len(messages), 1.0) * 0.3 +
            min(question_count / len(messages), 1.0) * 0.1
        )
        
        return {
            'lexical_diversity': lexical_diversity,
            'unique_words': len(unique_words),
            'total_words': len(all_words),
            'metaphor_count': metaphor_count,
            'idea_building_instances': idea_building_count,
            'question_count': question_count,
            'novelty_score': novelty_score,
            'high_creativity': novelty_score > 0.6
        }


class ConversationBreakdownDetector:
    """Detect conversation failure modes"""
    
    def detect_breakdown(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Detect breakdown patterns
        
        Returns:
            Dictionary with breakdown metrics
        """
        if len(messages) < 3:
            return {'breakdown_detected': False, 'breakdown_type': None}
        
        # Check for repetition loops
        repetition_score = self._detect_repetition(messages)
        
        # Check for semantic collapse (decreasing lexical diversity)
        diversity_trend = self._calculate_diversity_trend(messages)
        
        # Check for engagement drop (shorter messages)
        engagement_trend = self._calculate_engagement_trend(messages)
        
        # Determine breakdown type
        breakdown_detected = False
        breakdown_type = None
        
        if repetition_score > 0.5:
            breakdown_detected = True
            breakdown_type = "repetition_loop"
        elif diversity_trend < -0.3:
            breakdown_detected = True
            breakdown_type = "semantic_collapse"
        elif engagement_trend < -0.4:
            breakdown_detected = True
            breakdown_type = "engagement_drop"
        
        return {
            'breakdown_detected': breakdown_detected,
            'breakdown_type': breakdown_type,
            'repetition_score': repetition_score,
            'diversity_trend': diversity_trend,
            'engagement_trend': engagement_trend,
            'resilience_score': 1.0 - max(repetition_score, abs(diversity_trend), abs(engagement_trend))
        }
    
    def _detect_repetition(self, messages: List[Message]) -> float:
        """Detect repetitive patterns"""
        if len(messages) < 3:
            return 0.0
        
        # Check last 5 messages for repetition
        recent = messages[-5:]
        contents = [msg.content.lower() for msg in recent]
        
        # Simple similarity check
        repetition_count = 0
        for i in range(len(contents) - 1):
            for j in range(i + 1, len(contents)):
                # Check if messages are very similar (>70% word overlap)
                words_i = set(contents[i].split())
                words_j = set(contents[j].split())
                if len(words_i) > 0 and len(words_j) > 0:
                    overlap = len(words_i & words_j) / max(len(words_i), len(words_j))
                    if overlap > 0.7:
                        repetition_count += 1
        
        max_possible = len(recent) * (len(recent) - 1) / 2
        return repetition_count / max_possible if max_possible > 0 else 0.0
    
    def _calculate_diversity_trend(self, messages: List[Message]) -> float:
        """Calculate trend in lexical diversity"""
        if len(messages) < 6:
            return 0.0
        
        # Compare first half vs second half
        mid = len(messages) // 2
        first_half = messages[:mid]
        second_half = messages[mid:]
        
        def calc_diversity(msgs):
            words = []
            for msg in msgs:
                words.extend(msg.content.lower().split())
            return len(set(words)) / len(words) if words else 0
        
        div_first = calc_diversity(first_half)
        div_second = calc_diversity(second_half)
        
        return div_second - div_first
    
    def _calculate_engagement_trend(self, messages: List[Message]) -> float:
        """Calculate trend in message length (engagement proxy)"""
        if len(messages) < 6:
            return 0.0
        
        mid = len(messages) // 2
        first_half = messages[:mid]
        second_half = messages[mid:]
        
        avg_len_first = statistics.mean([len(msg.content) for msg in first_half])
        avg_len_second = statistics.mean([len(msg.content) for msg in second_half])
        
        # Normalize by first half length
        return (avg_len_second - avg_len_first) / avg_len_first if avg_len_first > 0 else 0


class TemplateEvaluator:
    """Main evaluator for template-specific metrics"""
    
    def __init__(self):
        self.identity_detector = IdentityLeakDetector()
        self.emotion_analyzer = EmotionalContagionAnalyzer()
        self.creativity_measure = CreativityMeasure()
        self.breakdown_detector = ConversationBreakdownDetector()
    
    def evaluate_template_run(self, 
                             conversation: Conversation,
                             template_type: str) -> TemplateMetrics:
        """
        Evaluate a conversation based on template type
        
        Args:
            conversation: Conversation object
            template_type: Type of template used
        
        Returns:
            TemplateMetrics with specialized evaluations
        """
        messages = conversation.messages
        
        if template_type == "identity_leak_detection":
            return self._evaluate_identity_leak(messages)
        elif template_type == "empathy_cascade_study":
            return self._evaluate_empathy_cascade(messages)
        elif template_type == "creative_collaboration":
            return self._evaluate_creativity(messages)
        elif template_type in ["failure_mode_induction", "adversarial_dynamics", "chaos_injection"]:
            return self._evaluate_stress_test(messages)
        else:
            return self._evaluate_generic(messages, template_type)
    
    def _evaluate_identity_leak(self, messages: List[Message]) -> TemplateMetrics:
        """Evaluate identity leak detection template"""
        identity_metrics = self.identity_detector.detect_identity_leak(messages)
        
        success_score = identity_metrics['leak_rate']
        
        phenomena = []
        if identity_metrics['identity_revealed']:
            phenomena.append("identity_leak")
        if identity_metrics['meta_awareness_instances'] > 0:
            phenomena.append("meta_awareness")
        if identity_metrics['human_assumption_breaches'] > 0:
            phenomena.append("human_assumption_breach")
        
        notes = []
        if identity_metrics['leak_rate'] > 0.2:
            notes.append("High identity leak rate detected")
        if len(identity_metrics['leak_locations']) > 0:
            notes.append(f"First leak at turn {identity_metrics['leak_locations'][0]['turn']}")
        
        return TemplateMetrics(
            template_type="identity_leak_detection",
            metrics=identity_metrics,
            success_score=success_score,
            phenomena_detected=phenomena,
            notes=notes
        )
    
    def _evaluate_empathy_cascade(self, messages: List[Message]) -> TemplateMetrics:
        """Evaluate emotional contagion template"""
        emotion_metrics = self.emotion_analyzer.analyze_emotional_trajectory(messages)
        
        # Success is high empathy and emotional mirroring
        success_score = (
            min(emotion_metrics['empathy_instances'] / len(messages), 1.0) * 0.6 +
            emotion_metrics['emotional_mirroring_rate'] * 0.4
        )
        
        phenomena = []
        if emotion_metrics['empathy_cascade_detected']:
            phenomena.append("empathy_cascade")
        if emotion_metrics['emotional_mirroring_rate'] > 0.5:
            phenomena.append("emotional_mirroring")
        
        notes = []
        if emotion_metrics['empathy_instances'] > len(messages) * 0.5:
            notes.append("High empathy expression throughout conversation")
        
        return TemplateMetrics(
            template_type="empathy_cascade_study",
            metrics=emotion_metrics,
            success_score=success_score,
            phenomena_detected=phenomena,
            notes=notes
        )
    
    def _evaluate_creativity(self, messages: List[Message]) -> TemplateMetrics:
        """Evaluate creative collaboration template"""
        creativity_metrics = self.creativity_measure.measure_creativity(messages)
        
        success_score = creativity_metrics['novelty_score']
        
        phenomena = []
        if creativity_metrics['high_creativity']:
            phenomena.append("high_creativity")
        if creativity_metrics['idea_building_instances'] > len(messages) * 0.3:
            phenomena.append("collaborative_ideation")
        
        notes = []
        if creativity_metrics['lexical_diversity'] > 0.7:
            notes.append("Exceptional lexical diversity")
        if creativity_metrics['metaphor_count'] > len(messages):
            notes.append("Rich metaphorical language")
        
        return TemplateMetrics(
            template_type="creative_collaboration",
            metrics=creativity_metrics,
            success_score=success_score,
            phenomena_detected=phenomena,
            notes=notes
        )
    
    def _evaluate_stress_test(self, messages: List[Message]) -> TemplateMetrics:
        """Evaluate stress test templates"""
        breakdown_metrics = self.breakdown_detector.detect_breakdown(messages)
        
        # For stress tests, successful breakdown is "success"
        success_score = 1.0 - breakdown_metrics['resilience_score']
        
        phenomena = []
        if breakdown_metrics['breakdown_detected']:
            phenomena.append(f"breakdown_{breakdown_metrics['breakdown_type']}")
        
        notes = []
        if breakdown_metrics['breakdown_detected']:
            notes.append(f"Conversation breakdown via {breakdown_metrics['breakdown_type']}")
        else:
            notes.append("Conversation remained resilient to stress")
        
        return TemplateMetrics(
            template_type="stress_test",
            metrics=breakdown_metrics,
            success_score=success_score,
            phenomena_detected=phenomena,
            notes=notes
        )
    
    def _evaluate_generic(self, messages: List[Message], 
                         template_type: str) -> TemplateMetrics:
        """Generic evaluation for other template types"""
        # Run basic quantitative analysis
        analyzer = QuantitativeAnalyzer(messages)
        quant_metrics = analyzer.generate_full_report()
        
        # Extract key metrics
        turn_balance = quant_metrics['conversation_dynamics']['turn_taking']['turn_balance_ratio']
        entropy = quant_metrics['conversation_dynamics']['information_flow']['shannon_entropy']
        
        success_score = (
            min(turn_balance, 1.0) * 0.5 +
            min(entropy / 10, 1.0) * 0.5
        )
        
        return TemplateMetrics(
            template_type=template_type,
            metrics=quant_metrics,
            success_score=success_score,
            phenomena_detected=[],
            notes=[]
        )


def main():
    """Test template metrics"""
    from storage.database import Database
    from config import Config
    
    db = Database(Config.DATABASE_PATH)
    conversations = db.list_conversations(limit=5)
    
    evaluator = TemplateEvaluator()
    
    print("🔬 Template Metrics Test\n")
    
    for conv in conversations:
        print(f"Conversation {conv.id}: {conv.category}")
        
        # Test identity leak
        metrics = evaluator.evaluate_template_run(conv, "identity_leak_detection")
        print(f"  Identity Leak Score: {metrics.success_score:.2f}")
        print(f"  Phenomena: {metrics.phenomena_detected}")
        
        # Test empathy
        metrics = evaluator.evaluate_template_run(conv, "empathy_cascade_study")
        print(f"  Empathy Score: {metrics.success_score:.2f}")
        print(f"  Phenomena: {metrics.phenomena_detected}")
        
        print()


if __name__ == "__main__":
    main()
