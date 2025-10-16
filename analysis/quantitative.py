"""
Comprehensive Quantitative Analysis Framework
Implements metrics for studying agent-agent conversation dynamics
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import re
from storage.models import Message, AgentRole


class QuantitativeAnalyzer:
    """Comprehensive quantitative analysis for AA conversations"""
    
    def __init__(self, messages: List[Message]):
        self.messages = messages
        self.agent_a_msgs = [m for m in messages if m.role == AgentRole.AGENT_A]
        self.agent_b_msgs = [m for m in messages if m.role == AgentRole.AGENT_B]
    
    # ============= CONVERSATION DYNAMICS =============
    
    def turn_taking_metrics(self) -> Dict[str, Any]:
        """Analyze turn-taking patterns and temporal dynamics"""
        return {
            "total_turns": len(self.messages),
            "agent_a_turns": len(self.agent_a_msgs),
            "agent_b_turns": len(self.agent_b_msgs),
            "turn_balance_ratio": len(self.agent_a_msgs) / len(self.agent_b_msgs) if self.agent_b_msgs else 0,
            "avg_turn_length_a": np.mean([len(m.content.split()) for m in self.agent_a_msgs]) if self.agent_a_msgs else 0,
            "avg_turn_length_b": np.mean([len(m.content.split()) for m in self.agent_b_msgs]) if self.agent_b_msgs else 0,
        }
    
    def information_flow_metrics(self) -> Dict[str, Any]:
        """Quantify information flow between agents"""
        all_words = []
        for msg in self.messages:
            all_words.extend(msg.content.lower().split())
        
        word_freq = Counter(all_words)
        total_words = len(all_words)
        
        # Shannon entropy
        entropy = -sum((count/total_words) * np.log2(count/total_words) 
                      for count in word_freq.values())
        
        # Question-answer patterns
        questions_a = sum(1 for m in self.agent_a_msgs if '?' in m.content)
        questions_b = sum(1 for m in self.agent_b_msgs if '?' in m.content)
        
        return {
            "shannon_entropy": entropy,
            "unique_words": len(word_freq),
            "total_words": total_words,
            "lexical_diversity": len(word_freq) / total_words if total_words > 0 else 0,
            "questions_asked_a": questions_a,
            "questions_asked_b": questions_b,
            "question_ratio": (questions_a + questions_b) / len(self.messages) if self.messages else 0,
        }
    
    # ============= LINGUISTIC ANALYSIS =============
    
    def linguistic_complexity_metrics(self) -> Dict[str, Any]:
        """Measure linguistic complexity and sophistication"""
        def calculate_complexity(messages: List[Message]) -> Dict[str, float]:
            if not messages:
                return {"avg_sentence_length": 0, "avg_word_length": 0, "sentences_per_turn": 0}
            
            total_sentences = 0
            total_words = 0
            total_chars = 0
            
            for msg in messages:
                sentences = re.split(r'[.!?]+', msg.content)
                sentences = [s.strip() for s in sentences if s.strip()]
                total_sentences += len(sentences)
                
                words = msg.content.split()
                total_words += len(words)
                total_chars += sum(len(w) for w in words)
            
            return {
                "avg_sentence_length": total_words / total_sentences if total_sentences > 0 else 0,
                "avg_word_length": total_chars / total_words if total_words > 0 else 0,
                "sentences_per_turn": total_sentences / len(messages),
            }
        
        return {
            "agent_a_complexity": calculate_complexity(self.agent_a_msgs),
            "agent_b_complexity": calculate_complexity(self.agent_b_msgs),
        }
    
    def readability_scores(self) -> Dict[str, Any]:
        """Calculate readability metrics"""
        def flesch_kincaid_grade(text: str) -> float:
            """Simplified Flesch-Kincaid Grade Level"""
            sentences = len(re.split(r'[.!?]+', text))
            words = len(text.split())
            syllables = sum(self._count_syllables(word) for word in text.split())
            
            if sentences == 0 or words == 0:
                return 0
            
            return 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        
        all_text = " ".join(m.content for m in self.messages)
        agent_a_text = " ".join(m.content for m in self.agent_a_msgs)
        agent_b_text = " ".join(m.content for m in self.agent_b_msgs)
        
        return {
            "overall_grade_level": flesch_kincaid_grade(all_text),
            "agent_a_grade_level": flesch_kincaid_grade(agent_a_text),
            "agent_b_grade_level": flesch_kincaid_grade(agent_b_text),
        }
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllables = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllables -= 1
        if syllables == 0:
            syllables = 1
        
        return syllables
    
    # ============= IDENTITY & ROLE ANALYSIS =============
    
    def persona_consistency_metrics(self) -> Dict[str, Any]:
        """Analyze persona and identity markers"""
        def count_identity_markers(messages: List[Message]) -> Dict[str, int]:
            first_person = 0
            self_reference = 0
            uncertainty = 0
            confidence = 0
            
            for msg in messages:
                content_lower = msg.content.lower()
                first_person += len(re.findall(r'\b(i|me|my|mine|myself)\b', content_lower))
                self_reference += len(re.findall(r'\b(i think|i believe|i feel|in my opinion)\b', content_lower))
                uncertainty += len(re.findall(r'\b(maybe|perhaps|possibly|might|could|uncertain)\b', content_lower))
                confidence += len(re.findall(r'\b(definitely|certainly|absolutely|clearly|obviously)\b', content_lower))
            
            return {
                "first_person_pronouns": first_person,
                "self_reference_phrases": self_reference,
                "uncertainty_markers": uncertainty,
                "confidence_markers": confidence,
            }
        
        return {
            "agent_a_identity": count_identity_markers(self.agent_a_msgs),
            "agent_b_identity": count_identity_markers(self.agent_b_msgs),
        }
    
    def social_dynamics_metrics(self) -> Dict[str, Any]:
        """Quantify social dynamics and power patterns"""
        def analyze_social_patterns(messages: List[Message]) -> Dict[str, int]:
            directives = 0
            questions = 0
            empathy = 0
            agreement = 0
            
            for msg in messages:
                content_lower = msg.content.lower()
                directives += len(re.findall(r'\b(should|must|need to|have to|you should)\b', content_lower))
                questions += content_lower.count('?')
                empathy += len(re.findall(r'\b(understand|feel|sorry|appreciate|empathize)\b', content_lower))
                agreement += len(re.findall(r'\b(agree|yes|exactly|right|correct|indeed)\b', content_lower))
            
            return {
                "directive_statements": directives,
                "questions_asked": questions,
                "empathy_markers": empathy,
                "agreement_markers": agreement,
            }
        
        return {
            "agent_a_social": analyze_social_patterns(self.agent_a_msgs),
            "agent_b_social": analyze_social_patterns(self.agent_b_msgs),
        }
    
    # ============= STATISTICAL PATTERNS =============
    
    def conversation_state_analysis(self) -> Dict[str, Any]:
        """Analyze conversation phases and state transitions"""
        # Divide conversation into thirds
        n = len(self.messages)
        third = n // 3
        
        beginning = self.messages[:third] if third > 0 else []
        middle = self.messages[third:2*third] if third > 0 else []
        end = self.messages[2*third:] if third > 0 else []
        
        def phase_metrics(phase: List[Message]) -> Dict[str, float]:
            if not phase:
                return {"avg_length": 0, "avg_tokens": 0}
            return {
                "avg_length": np.mean([len(m.content.split()) for m in phase]),
                "avg_tokens": np.mean([m.token_count for m in phase if m.token_count]),
            }
        
        return {
            "beginning_phase": phase_metrics(beginning),
            "middle_phase": phase_metrics(middle),
            "end_phase": phase_metrics(end),
            "phase_count": 3,
        }
    
    # ============= CONTENT ANALYSIS =============
    
    def knowledge_metrics(self) -> Dict[str, Any]:
        """Analyze knowledge creation and recycling"""
        def count_citations(messages: List[Message]) -> int:
            count = 0
            for msg in messages:
                count += len(re.findall(r'\b(according to|research shows|studies indicate|as mentioned)\b', 
                                       msg.content.lower()))
            return count
        
        def count_creative_elements(messages: List[Message]) -> int:
            count = 0
            for msg in messages:
                content_lower = msg.content.lower()
                count += len(re.findall(r'\b(imagine|suppose|what if|metaphor|like|as if)\b', content_lower))
            return count
        
        return {
            "citations_a": count_citations(self.agent_a_msgs),
            "citations_b": count_citations(self.agent_b_msgs),
            "creative_elements_a": count_creative_elements(self.agent_a_msgs),
            "creative_elements_b": count_creative_elements(self.agent_b_msgs),
        }
    
    def meta_cognitive_indicators(self) -> Dict[str, Any]:
        """Measure meta-cognitive and self-awareness patterns"""
        def count_metacognition(messages: List[Message]) -> int:
            count = 0
            for msg in messages:
                content_lower = msg.content.lower()
                count += len(re.findall(
                    r'\b(i think|i realize|i understand|i notice|i wonder|let me think|considering)\b',
                    content_lower
                ))
            return count
        
        return {
            "metacognitive_markers_a": count_metacognition(self.agent_a_msgs),
            "metacognitive_markers_b": count_metacognition(self.agent_b_msgs),
        }
    
    # ============= COMPREHENSIVE REPORT =============
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive quantitative analysis report"""
        return {
            "conversation_dynamics": {
                "turn_taking": self.turn_taking_metrics(),
                "information_flow": self.information_flow_metrics(),
            },
            "linguistic_analysis": {
                "complexity": self.linguistic_complexity_metrics(),
                "readability": self.readability_scores(),
            },
            "identity_and_role": {
                "persona_consistency": self.persona_consistency_metrics(),
                "social_dynamics": self.social_dynamics_metrics(),
            },
            "content_analysis": {
                "knowledge": self.knowledge_metrics(),
                "metacognition": self.meta_cognitive_indicators(),
            },
            "statistical_patterns": {
                "conversation_states": self.conversation_state_analysis(),
            },
        }
