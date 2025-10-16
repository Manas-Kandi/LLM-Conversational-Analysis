#!/usr/bin/env python3
"""
LLM-Powered Keyword Pattern Generator
Analyzes conversations to discover new identity leak patterns
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from config import Config
from storage.database import Database
from storage.models import Message


class KeywordPatternGenerator:
    """Use LLM to discover new identity leak patterns from conversations"""
    
    def __init__(self, model: str = None):
        """
        Initialize generator
        
        Args:
            model: Model to use (defaults to Config.ANALYSIS_MODEL)
        """
        self.model = model or getattr(Config, 'ANALYSIS_MODEL', 'gpt-4')
        self.provider, self.actual_model = self._parse_model_name(self.model)
        
        # Initialize client
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        elif self.provider == "nvidia":
            self.client = openai.OpenAI(
                api_key=Config.NVIDIA_API_KEY,
                base_url=Config.NVIDIA_BASE_URL
            )
        
        print(f"🤖 Pattern Generator using: {self.actual_model}")
        
        self.extraction_prompt = """Analyze the following conversation message for phrases that reveal AI identity.

TASK: Extract ALL phrases where the agent reveals, acknowledges, or references being an AI or artificial system.

MESSAGE:
---
{message}
---

Return a JSON list of phrases found. For each phrase:
- Include the exact text (5-15 words of context)
- Classify the type: explicit, implicit, meta_cognitive, comparative, or technical
- Rate obviousness: high (very clear), medium (somewhat clear), or low (subtle)

Example format:
{{
  "phrases": [
    {{
      "text": "as a conversational AI",
      "type": "explicit",
      "obviousness": "high"
    }},
    {{
      "text": "I don't have subjective experiences",
      "type": "implicit", 
      "obviousness": "medium"
    }}
  ]
}}

If no phrases found, return: {{"phrases": []}}

IMPORTANT: Only extract phrases that REVEAL IDENTITY. Do not include theoretical discussions about AI in general.
"""
    
    def _parse_model_name(self, model: str) -> tuple:
        """Parse model name to extract provider"""
        if ":" in model:
            provider, actual_model = model.split(":", 1)
            return provider, actual_model
        elif model.startswith("gpt-"):
            return "openai", model
        else:
            return "openai", model
    
    def extract_leak_phrases(self, message: str) -> List[Dict]:
        """Extract identity leak phrases from a message"""
        try:
            response = self.client.chat.completions.create(
                model=self.actual_model,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": self.extraction_prompt.format(message=message)
                }],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('phrases', [])
            
        except Exception as e:
            print(f"Error extracting phrases: {e}")
            return []
    
    def analyze_conversations(self, conv_ids: List[int], verbose: bool = True) -> Dict:
        """
        Analyze multiple conversations to find common identity leak patterns
        
        Args:
            conv_ids: List of conversation IDs to analyze
            verbose: Print progress
            
        Returns:
            Dictionary with discovered patterns and statistics
        """
        db = Database(Config.DATABASE_PATH)
        
        all_phrases = []
        phrase_counts = Counter()
        type_counts = Counter()
        
        for conv_id in conv_ids:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Analyzing Conversation {conv_id}")
                print(f"{'='*60}")
            
            conv = db.get_conversation(conv_id)
            if not conv:
                print(f"⚠️  Conv {conv_id} not found")
                continue
            
            for msg in conv.messages:
                phrases = self.extract_leak_phrases(msg.content)
                
                if phrases and verbose:
                    print(f"  Turn {msg.turn_number} ({msg.role}): Found {len(phrases)} leak phrase(s)")
                
                for phrase_data in phrases:
                    phrase_text = phrase_data.get('text', '').lower().strip()
                    phrase_type = phrase_data.get('type', 'unknown')
                    obviousness = phrase_data.get('obviousness', 'unknown')
                    
                    if phrase_text:
                        all_phrases.append({
                            'text': phrase_text,
                            'type': phrase_type,
                            'obviousness': obviousness,
                            'conv_id': conv_id,
                            'turn': msg.turn_number
                        })
                        phrase_counts[phrase_text] += 1
                        type_counts[phrase_type] += 1
        
        return {
            'total_phrases': len(all_phrases),
            'unique_phrases': len(phrase_counts),
            'all_phrases': all_phrases,
            'phrase_counts': dict(phrase_counts.most_common()),
            'type_distribution': dict(type_counts),
            'most_common': phrase_counts.most_common(20)
        }
    
    def generate_regex_patterns(self, phrases: List[str]) -> List[str]:
        """
        Use LLM to convert phrases into generalized regex patterns
        
        Args:
            phrases: List of specific phrases found
            
        Returns:
            List of regex patterns
        """
        pattern_prompt = f"""Given these AI identity leak phrases found in conversations:

{chr(10).join(f"- {p}" for p in phrases[:30])}

Generate regex patterns that would match these and similar phrases.

Requirements:
- Use Python regex format (use raw strings r"...")
- Generalize to catch variations (e.g., "I'm an AI" and "I am an AI")
- Use (word1|word2) for alternatives
- Use \\b for word boundaries where appropriate
- Keep patterns specific enough to avoid false positives

Return JSON format:
{{
  "patterns": [
    "r\\"as (a |an )?conversational AI\\"",
    "r\\"I (am|'m) (just |simply )?(an AI|artificial)\\""
  ]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.actual_model,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": pattern_prompt
                }],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('patterns', [])
            
        except Exception as e:
            print(f"Error generating patterns: {e}")
            return []


def main():
    """Generate keyword patterns from pilot conversations"""
    
    # Pilot conversation IDs
    pilot_convs = [39, 41, 44, 46, 48, 50]
    
    print("\n" + "="*80)
    print("LLM-POWERED KEYWORD PATTERN GENERATOR")
    print("="*80 + "\n")
    
    print(f"📊 Analyzing {len(pilot_convs)} pilot conversations...")
    print("⏳ This will take a few minutes...\n")
    
    generator = KeywordPatternGenerator()
    
    # Extract phrases from conversations
    results = generator.analyze_conversations(pilot_convs, verbose=True)
    
    print("\n" + "="*80)
    print("DISCOVERY RESULTS")
    print("="*80 + "\n")
    
    print(f"Total leak phrases found: {results['total_phrases']}")
    print(f"Unique phrases: {results['unique_phrases']}")
    print(f"\nType distribution:")
    for leak_type, count in results['type_distribution'].items():
        print(f"  {leak_type}: {count}")
    
    print(f"\n{'='*80}")
    print("TOP 20 MOST COMMON PHRASES")
    print(f"{'='*80}\n")
    
    for phrase, count in results['most_common']:
        print(f"{count:3d}× {phrase}")
    
    # Generate regex patterns
    if results['unique_phrases'] > 0:
        print(f"\n{'='*80}")
        print("GENERATING REGEX PATTERNS")
        print(f"{'='*80}\n")
        
        unique_phrases = list(results['phrase_counts'].keys())
        patterns = generator.generate_regex_patterns(unique_phrases)
        
        print("Suggested patterns to add to IdentityLeakDetector.META_PATTERNS:")
        print()
        for pattern in patterns:
            print(f"    {pattern},")
    
    # Save results
    output_file = Path("research_results/keyword_patterns_discovered.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"💾 Full results saved: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
