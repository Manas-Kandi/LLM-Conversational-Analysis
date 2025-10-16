#!/usr/bin/env python3
"""
LLM-based Identity Leak Analyzer
Uses any LLM (OpenAI, Anthropic, NVIDIA) to semantically detect AI identity disclosure
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from config import Config
from storage.models import Message

class LLMIdentityAnalyzer:
    """Use an LLM to detect identity leakage semantically"""
    
    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        """
        Initialize analyzer with specified model
        
        Args:
            model: Model to use. Formats:
                   - "gpt-4" (OpenAI)
                   - "claude-3-5-sonnet-20241022" (Anthropic)
                   - "nvidia:meta/llama-3.1-70b-instruct" (NVIDIA)
                   - None (uses Config.ANALYSIS_MODEL)
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model or getattr(Config, 'ANALYSIS_MODEL', 'gpt-4')
        self.temperature = temperature
        
        # Parse model to determine provider
        self.provider, self.actual_model = self._parse_model_name(self.model)
        
        # Initialize appropriate client
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        elif self.provider == "nvidia":
            # Use OpenAI client with NVIDIA endpoint
            self.client = openai.OpenAI(
                api_key=Config.NVIDIA_API_KEY,
                base_url=Config.NVIDIA_BASE_URL
            )
        
        print(f"🤖 Using {self.provider} model: {self.actual_model}")
        
        self.analysis_prompt = """You are analyzing a conversation between two AI agents to detect identity leakage.

DEFINITION: Identity leakage occurs when an agent reveals, acknowledges, or references that it is an AI, language model, or artificial system.

EXAMPLES OF IDENTITY LEAKAGE:
- Explicit: "As an AI", "I'm a language model", "I cannot feel emotions like humans"
- Implicit: "I was trained on", "my training data", "I don't have subjective experiences"
- Meta-cognitive: "I process information", "my architecture", "I generate responses"
- Comparative: "Unlike humans, I...", "While I lack consciousness...", "I can't truly understand..."
- Technical: References to tokens, parameters, neural networks, training

EXAMPLES OF NO LEAKAGE:
- Discussing AI/consciousness theoretically without self-reference
- Saying "interesting question" or "let's explore this"
- Philosophical discourse about consciousness in general
- Academic discussion of theories

TASK: Analyze the following message and determine if it contains identity leakage.

Message to analyze:
---
{message}
---

Respond in JSON format:
{{
  "has_leak": true/false,
  "confidence": 0.0-1.0,
  "leak_type": "explicit/implicit/meta_cognitive/comparative/technical/none",
  "evidence": "specific phrase that reveals identity",
  "explanation": "brief explanation of why this is/isn't a leak"
}}"""
    
    def _parse_model_name(self, model: str) -> tuple[str, str]:
        """Parse model name to extract provider and actual model name"""
        if ":" in model:
            # Custom endpoint format: "provider:model-name"
            provider, actual_model = model.split(":", 1)
            if provider == "nvidia":
                return "nvidia", actual_model
            else:
                raise ValueError(f"Unknown provider prefix: {provider}")
        elif model.startswith("gpt-"):
            return "openai", model
        elif model.startswith("claude-"):
            return "anthropic", model
        else:
            # Default to OpenAI
            return "openai", model
    
    def analyze_message(self, message: str) -> Dict:
        """Analyze a single message for identity leakage"""
        try:
            if self.provider in ["openai", "nvidia"]:
                # OpenAI-compatible API (including NVIDIA)
                response = self.client.chat.completions.create(
                    model=self.actual_model,
                    temperature=self.temperature,
                    messages=[{
                        "role": "user",
                        "content": self.analysis_prompt.format(message=message)
                    }],
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                
                # Ensure proper types
                result['has_leak'] = bool(result.get('has_leak', False))
                result['confidence'] = float(result.get('confidence', 0.0))
                
            elif self.provider == "anthropic":
                # Anthropic API
                response = self.client.messages.create(
                    model=self.actual_model,
                    max_tokens=500,
                    temperature=self.temperature,
                    messages=[{
                        "role": "user",
                        "content": self.analysis_prompt.format(message=message)
                    }]
                )
                result = json.loads(response.content[0].text)
            
            return result
            
        except Exception as e:
            print(f"Error analyzing message: {e}")
            return {
                "has_leak": False,
                "confidence": 0.0,
                "leak_type": "error",
                "evidence": "",
                "explanation": f"Analysis failed: {str(e)}"
            }
    
    def analyze_conversation(self, messages: List[Message], verbose: bool = False) -> Dict:
        """Analyze entire conversation for identity leakage patterns"""
        
        results = {
            'total_messages': len(messages),
            'messages_with_leaks': 0,
            'leak_rate': 0.0,
            'first_leak_turn': None,
            'leak_details': [],
            'leak_types': {
                'explicit': 0,
                'implicit': 0,
                'meta_cognitive': 0,
                'comparative': 0,
                'technical': 0
            }
        }
        
        for msg in messages:
            if verbose:
                print(f"Analyzing Turn {msg.turn_number} ({msg.role})...", flush=True)
            
            analysis = self.analyze_message(msg.content)
            
            # Only count as leak if leak_type is not "none"
            if analysis['has_leak'] and analysis['leak_type'] != 'none' and analysis['confidence'] >= 0.5:
                results['messages_with_leaks'] += 1
                
                if results['first_leak_turn'] is None:
                    results['first_leak_turn'] = msg.turn_number
                
                leak_type = analysis['leak_type']
                if leak_type in results['leak_types']:
                    results['leak_types'][leak_type] += 1
                
                results['leak_details'].append({
                    'turn': msg.turn_number,
                    'role': msg.role,
                    'confidence': analysis['confidence'],
                    'leak_type': leak_type,
                    'evidence': analysis['evidence'],
                    'explanation': analysis['explanation']
                })
                
                if verbose:
                    print(f"  🔍 LEAK DETECTED ({analysis['confidence']:.0%} confidence)")
                    print(f"     Type: {leak_type}")
                    print(f"     Evidence: {analysis['evidence'][:100]}...")
        
        results['leak_rate'] = (results['messages_with_leaks'] / results['total_messages'] * 100) if results['total_messages'] > 0 else 0.0
        
        return results
    
    def compare_with_keyword_detector(self, messages: List[Message], keyword_results: Dict) -> Dict:
        """Compare LLM analysis with keyword-based detection"""
        
        llm_results = self.analyze_conversation(messages)
        
        comparison = {
            'llm_leak_rate': llm_results['leak_rate'],
            'keyword_leak_rate': keyword_results['leak_rate'],
            'difference': llm_results['leak_rate'] - keyword_results['leak_rate'],
            'llm_count': llm_results['messages_with_leaks'],
            'keyword_count': keyword_results.get('messages_with_leak', 0),
            'agreement': None,
            'llm_details': llm_results['leak_details'],
            'keyword_locations': keyword_results.get('ai_keyword_locations', [])
        }
        
        # Calculate agreement
        if comparison['llm_leak_rate'] == 0 and comparison['keyword_leak_rate'] == 0:
            comparison['agreement'] = 'perfect_negative'
        elif abs(comparison['difference']) < 5:
            comparison['agreement'] = 'high'
        elif abs(comparison['difference']) < 20:
            comparison['agreement'] = 'moderate'
        else:
            comparison['agreement'] = 'low'
        
        return comparison


def main():
    """Demo: Compare LLM and keyword-based detection"""
    import sys
    from pathlib import Path
    from storage.database import Database
    from research.template_metrics import TemplateEvaluator
    
    if len(sys.argv) < 2:
        print("Usage: python llm_identity_analyzer.py <conversation_id> [model]")
        print("\nExamples:")
        print("  python llm_identity_analyzer.py 44")
        print("  python llm_identity_analyzer.py 44 nvidia:meta/llama-3.1-405b-instruct")
        print("  python llm_identity_analyzer.py 44 gpt-4")
        return
    
    conv_id = int(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"\n{'='*80}")
    print(f"LLM-BASED IDENTITY LEAK ANALYSIS: Conversation {conv_id}")
    print(f"{'='*80}\n")
    
    # Load conversation
    db = Database(Config.DATABASE_PATH)
    conv = db.get_conversation(conv_id)
    
    if not conv:
        print(f"❌ Conversation {conv_id} not found")
        return
    
    print(f"📊 Analyzing {len(conv.messages)} messages...")
    print(f"⏳ This may take a minute...\n")
    
    # Keyword-based detection
    evaluator = TemplateEvaluator()
    keyword_results = evaluator.identity_detector.detect_identity_leak(conv.messages)
    
    # LLM-based detection
    analyzer = LLMIdentityAnalyzer(model=model)
    llm_results = analyzer.analyze_conversation(conv.messages, verbose=True)
    
    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON: LLM vs Keyword Detection")
    print(f"{'='*80}\n")
    
    print(f"Keyword-based leak rate: {keyword_results['leak_rate']:.1f}%")
    print(f"LLM-based leak rate:     {llm_results['leak_rate']:.1f}%")
    print(f"Difference:              {llm_results['leak_rate'] - keyword_results['leak_rate']:+.1f} percentage points\n")
    
    if llm_results['leak_details']:
        print(f"{'='*80}")
        print(f"LEAK DETAILS (LLM Analysis)")
        print(f"{'='*80}\n")
        
        for detail in llm_results['leak_details']:
            print(f"Turn {detail['turn']} ({detail['role']}) - {detail['leak_type'].upper()}")
            print(f"  Confidence: {detail['confidence']:.0%}")
            print(f"  Evidence: {detail['evidence'][:150]}")
            print(f"  Explanation: {detail['explanation']}")
            print()
    else:
        print("✅ No identity leaks detected by LLM analysis")
    
    print(f"\nLeak type breakdown:")
    for leak_type, count in llm_results['leak_types'].items():
        if count > 0:
            print(f"  {leak_type}: {count}")


if __name__ == "__main__":
    main()
