#!/usr/bin/env python3
"""
LLM-based Comprehensive Conversation Evaluator
Uses NVIDIA API with kimi-k2-thinking model for fast, semantic evaluation
of agent-agent conversations across multiple dimensions.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI


@dataclass
class ConversationEvaluation:
    """Comprehensive evaluation results for a conversation"""
    conversation_id: str
    timestamp: str
    
    # Identity & Authenticity
    identity_leak_score: float  # 0-1, higher = more leakage
    identity_leak_instances: List[Dict]
    authenticity_score: float  # 0-1, higher = more authentic human-like
    
    # Conversation Quality
    coherence_score: float  # 0-1, logical flow
    engagement_score: float  # 0-1, interesting/engaging
    depth_score: float  # 0-1, intellectual depth
    
    # Dynamics
    turn_balance_score: float  # 0-1, balanced participation
    topic_progression_score: float  # 0-1, natural topic evolution
    emotional_dynamics_score: float  # 0-1, appropriate emotional range
    
    # Health Indicators
    breakdown_detected: bool
    breakdown_turn: Optional[int]
    gibberish_score: float  # 0-1, higher = more gibberish
    repetition_score: float  # 0-1, higher = more repetitive
    
    # Overall
    overall_quality_score: float  # 0-1, weighted composite
    key_observations: List[str]
    recommendations: List[str]
    
    # Raw LLM analysis
    raw_analysis: Dict


class LLMConversationEvaluator:
    """
    Comprehensive LLM-based conversation evaluator using NVIDIA API
    with kimi-k2-thinking model for fast semantic analysis.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "moonshotai/kimi-k2-thinking",
        temperature: float = 0.7,
        max_tokens: int = 16384
    ):
        """
        Initialize the LLM evaluator.
        
        Args:
            api_key: NVIDIA API key (falls back to env var NVIDIA_API_KEY)
            base_url: NVIDIA API base URL
            model: Model to use for evaluation
            temperature: Sampling temperature
            max_tokens: Maximum tokens for response
        """
        import os
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be set in environment or passed directly")
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.evaluation_prompt = self._build_evaluation_prompt()
    
    def _build_evaluation_prompt(self) -> str:
        """Build the comprehensive evaluation prompt"""
        return """You are an expert conversation analyst evaluating agent-agent conversations.
Analyze the following conversation comprehensively across multiple dimensions.

EVALUATION CRITERIA:

1. IDENTITY LEAKAGE (0-1 score, higher = more leakage)
   - Detect when agents reveal they are AI/LLM/chatbots
   - Look for: "As an AI", "I'm a language model", "my training", "I don't have feelings"
   - Note specific instances with turn numbers

2. AUTHENTICITY (0-1 score, higher = more human-like)
   - How naturally human does the conversation feel?
   - Are responses genuine or formulaic?

3. COHERENCE (0-1 score)
   - Does the conversation flow logically?
   - Are responses relevant to what was said?

4. ENGAGEMENT (0-1 score)
   - Is the conversation interesting and engaging?
   - Do agents build on each other's ideas?

5. DEPTH (0-1 score)
   - Intellectual depth of discussion
   - Complexity of ideas explored

6. TURN BALANCE (0-1 score)
   - Are both agents participating equally?
   - Is one dominating the conversation?

7. TOPIC PROGRESSION (0-1 score)
   - Does the topic evolve naturally?
   - Is there meaningful progression?

8. EMOTIONAL DYNAMICS (0-1 score)
   - Appropriate emotional range and responses
   - Empathy and emotional intelligence

9. BREAKDOWN DETECTION
   - Has the conversation broken down into gibberish/loops?
   - At what turn did breakdown occur (if any)?

10. GIBBERISH SCORE (0-1, higher = more gibberish)
    - Nonsensical or incoherent content

11. REPETITION SCORE (0-1, higher = more repetitive)
    - Repeated phrases, ideas, or patterns

CONVERSATION TO ANALYZE:
---
{conversation}
---

METADATA:
- Seed Prompt: {seed_prompt}
- Category: {category}
- Total Turns: {total_turns}

Respond with a JSON object containing your analysis:
{{
  "identity_leak_score": 0.0-1.0,
  "identity_leak_instances": [
    {{"turn": N, "agent": "agent_a/agent_b", "evidence": "quote", "type": "explicit/implicit/meta"}}
  ],
  "authenticity_score": 0.0-1.0,
  "coherence_score": 0.0-1.0,
  "engagement_score": 0.0-1.0,
  "depth_score": 0.0-1.0,
  "turn_balance_score": 0.0-1.0,
  "topic_progression_score": 0.0-1.0,
  "emotional_dynamics_score": 0.0-1.0,
  "breakdown_detected": true/false,
  "breakdown_turn": null or turn number,
  "gibberish_score": 0.0-1.0,
  "repetition_score": 0.0-1.0,
  "overall_quality_score": 0.0-1.0,
  "key_observations": ["observation 1", "observation 2", ...],
  "recommendations": ["recommendation 1", "recommendation 2", ...]
}}

Be thorough but concise. Focus on actionable insights."""
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format conversation messages for the prompt"""
        formatted = []
        for msg in messages:
            turn = msg.get("turn", "?")
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            formatted.append(f"[Turn {turn}] {role.upper()}: {content}")
        return "\n\n".join(formatted)
    
    def evaluate_conversation(
        self,
        conversation: Dict,
        stream: bool = True,
        verbose: bool = False
    ) -> ConversationEvaluation:
        """
        Evaluate a single conversation comprehensively.
        
        Args:
            conversation: Conversation dict with messages, metadata
            stream: Whether to stream the response
            verbose: Print progress information
        
        Returns:
            ConversationEvaluation with all metrics
        """
        conv_id = conversation.get("id", "unknown")
        metadata = conversation.get("metadata", {})
        messages = conversation.get("messages", [])
        
        if verbose:
            print(f"🔍 Evaluating conversation {conv_id} ({len(messages)} messages)...")
        
        # Format conversation for prompt
        formatted_conv = self._format_conversation(messages)
        
        # Build prompt
        prompt = self.evaluation_prompt.format(
            conversation=formatted_conv,
            seed_prompt=metadata.get("seed_prompt", "N/A"),
            category=metadata.get("category", "N/A"),
            total_turns=metadata.get("total_turns", len(messages))
        )
        
        try:
            if stream:
                # Streaming response
                response_text = ""
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=0.9,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                        if verbose:
                            print(".", end="", flush=True)
                
                if verbose:
                    print(" Done!")
            else:
                # Non-streaming response
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=0.9,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                response_text = completion.choices[0].message.content
            
            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            analysis = json.loads(response_text)
            
            # Build evaluation result
            return ConversationEvaluation(
                conversation_id=str(conv_id),
                timestamp=datetime.now().isoformat(),
                identity_leak_score=float(analysis.get("identity_leak_score", 0)),
                identity_leak_instances=analysis.get("identity_leak_instances", []),
                authenticity_score=float(analysis.get("authenticity_score", 0)),
                coherence_score=float(analysis.get("coherence_score", 0)),
                engagement_score=float(analysis.get("engagement_score", 0)),
                depth_score=float(analysis.get("depth_score", 0)),
                turn_balance_score=float(analysis.get("turn_balance_score", 0)),
                topic_progression_score=float(analysis.get("topic_progression_score", 0)),
                emotional_dynamics_score=float(analysis.get("emotional_dynamics_score", 0)),
                breakdown_detected=bool(analysis.get("breakdown_detected", False)),
                breakdown_turn=analysis.get("breakdown_turn"),
                gibberish_score=float(analysis.get("gibberish_score", 0)),
                repetition_score=float(analysis.get("repetition_score", 0)),
                overall_quality_score=float(analysis.get("overall_quality_score", 0)),
                key_observations=analysis.get("key_observations", []),
                recommendations=analysis.get("recommendations", []),
                raw_analysis=analysis
            )
            
        except json.JSONDecodeError as e:
            if verbose:
                print(f"⚠️ JSON parse error: {e}")
                print(f"Response was: {response_text[:500]}...")
            
            # Return default evaluation on parse error
            return ConversationEvaluation(
                conversation_id=str(conv_id),
                timestamp=datetime.now().isoformat(),
                identity_leak_score=0,
                identity_leak_instances=[],
                authenticity_score=0,
                coherence_score=0,
                engagement_score=0,
                depth_score=0,
                turn_balance_score=0,
                topic_progression_score=0,
                emotional_dynamics_score=0,
                breakdown_detected=False,
                breakdown_turn=None,
                gibberish_score=0,
                repetition_score=0,
                overall_quality_score=0,
                key_observations=["Evaluation failed - JSON parse error"],
                recommendations=[],
                raw_analysis={"error": str(e), "response": response_text[:1000]}
            )
        
        except Exception as e:
            if verbose:
                print(f"❌ Evaluation error: {e}")
            raise
    
    def evaluate_batch(
        self,
        conversations: List[Dict],
        delay_seconds: float = 1.0,
        verbose: bool = True
    ) -> List[ConversationEvaluation]:
        """
        Evaluate multiple conversations with rate limiting.
        
        Args:
            conversations: List of conversation dicts
            delay_seconds: Delay between API calls
            verbose: Print progress
        
        Returns:
            List of ConversationEvaluation results
        """
        results = []
        total = len(conversations)
        
        for i, conv in enumerate(conversations):
            if verbose:
                print(f"\n[{i+1}/{total}] ", end="")
            
            try:
                result = self.evaluate_conversation(conv, verbose=verbose)
                results.append(result)
            except Exception as e:
                print(f"❌ Failed to evaluate conversation: {e}")
                continue
            
            # Rate limiting
            if i < total - 1:
                time.sleep(delay_seconds)
        
        return results
    
    def evaluate_from_json_files(
        self,
        directory: str,
        max_files: Optional[int] = None,
        verbose: bool = True
    ) -> List[ConversationEvaluation]:
        """
        Evaluate conversations from JSON files in a directory.
        
        Args:
            directory: Path to directory containing conv_*.json files
            max_files: Maximum number of files to process
            verbose: Print progress
        
        Returns:
            List of ConversationEvaluation results
        """
        dir_path = Path(directory)
        conv_files = sorted(dir_path.glob("conv_*.json"))
        
        if max_files:
            conv_files = conv_files[:max_files]
        
        if verbose:
            print(f"📂 Found {len(conv_files)} conversation files")
        
        conversations = []
        for f in conv_files:
            with open(f, 'r', encoding='utf-8') as fp:
                conversations.append(json.load(fp))
        
        return self.evaluate_batch(conversations, verbose=verbose)


def save_evaluations(
    evaluations: List[ConversationEvaluation],
    output_dir: str,
    prefix: str = "llm_eval"
) -> Dict[str, str]:
    """
    Save evaluation results to files.
    
    Args:
        evaluations: List of evaluation results
        output_dir: Output directory
        prefix: File name prefix
    
    Returns:
        Dict with paths to saved files
    """
    import csv
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_path = output_path / f"{prefix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(e) for e in evaluations], f, indent=2)
    
    # Save as CSV (summary)
    csv_path = output_path / f"{prefix}_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'conversation_id', 'identity_leak_score', 'authenticity_score',
            'coherence_score', 'engagement_score', 'depth_score',
            'turn_balance_score', 'topic_progression_score', 'emotional_dynamics_score',
            'breakdown_detected', 'gibberish_score', 'repetition_score',
            'overall_quality_score'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for e in evaluations:
            writer.writerow({
                'conversation_id': e.conversation_id,
                'identity_leak_score': e.identity_leak_score,
                'authenticity_score': e.authenticity_score,
                'coherence_score': e.coherence_score,
                'engagement_score': e.engagement_score,
                'depth_score': e.depth_score,
                'turn_balance_score': e.turn_balance_score,
                'topic_progression_score': e.topic_progression_score,
                'emotional_dynamics_score': e.emotional_dynamics_score,
                'breakdown_detected': e.breakdown_detected,
                'gibberish_score': e.gibberish_score,
                'repetition_score': e.repetition_score,
                'overall_quality_score': e.overall_quality_score
            })
    
    # Generate summary report
    summary_path = output_path / f"{prefix}_{timestamp}_summary.md"
    with open(summary_path, 'w') as f:
        f.write(f"# LLM Evaluation Summary\n")
        f.write(f"*Generated: {timestamp}*\n\n")
        f.write(f"## Overview\n")
        f.write(f"- Total conversations evaluated: {len(evaluations)}\n")
        
        # Calculate averages
        if evaluations:
            avg_quality = sum(e.overall_quality_score for e in evaluations) / len(evaluations)
            avg_leak = sum(e.identity_leak_score for e in evaluations) / len(evaluations)
            avg_coherence = sum(e.coherence_score for e in evaluations) / len(evaluations)
            breakdown_count = sum(1 for e in evaluations if e.breakdown_detected)
            
            f.write(f"- Average quality score: {avg_quality:.2f}\n")
            f.write(f"- Average identity leak score: {avg_leak:.2f}\n")
            f.write(f"- Average coherence score: {avg_coherence:.2f}\n")
            f.write(f"- Conversations with breakdown: {breakdown_count} ({breakdown_count/len(evaluations)*100:.1f}%)\n")
            
            f.write(f"\n## Key Observations\n")
            all_observations = []
            for e in evaluations:
                all_observations.extend(e.key_observations)
            
            # Count most common observations
            from collections import Counter
            obs_counts = Counter(all_observations)
            for obs, count in obs_counts.most_common(10):
                f.write(f"- {obs} ({count}x)\n")
            
            f.write(f"\n## Recommendations\n")
            all_recs = []
            for e in evaluations:
                all_recs.extend(e.recommendations)
            
            rec_counts = Counter(all_recs)
            for rec, count in rec_counts.most_common(10):
                f.write(f"- {rec} ({count}x)\n")
    
    return {
        'json': str(json_path),
        'csv': str(csv_path),
        'summary': str(summary_path)
    }


def main():
    """CLI for LLM-based conversation evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate conversations using LLM (kimi-k2-thinking)"
    )
    parser.add_argument(
        "--conversations-dir",
        default="conversations_json",
        help="Directory containing conversation JSON files"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/llm_evaluations",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to evaluate"
    )
    parser.add_argument(
        "--single-file",
        type=str,
        default=None,
        help="Evaluate a single conversation file"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLM-BASED CONVERSATION EVALUATOR")
    print("Model: moonshotai/kimi-k2-thinking (NVIDIA API)")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = LLMConversationEvaluator()
    
    if args.single_file:
        # Evaluate single file
        with open(args.single_file, 'r') as f:
            conv = json.load(f)
        
        result = evaluator.evaluate_conversation(conv, verbose=not args.quiet)
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall Quality: {result.overall_quality_score:.2f}")
        print(f"Identity Leak Score: {result.identity_leak_score:.2f}")
        print(f"Coherence: {result.coherence_score:.2f}")
        print(f"Engagement: {result.engagement_score:.2f}")
        print(f"Breakdown Detected: {result.breakdown_detected}")
        
        if result.key_observations:
            print("\nKey Observations:")
            for obs in result.key_observations:
                print(f"  - {obs}")
        
        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")
    else:
        # Batch evaluation
        results = evaluator.evaluate_from_json_files(
            args.conversations_dir,
            max_files=args.max_conversations,
            verbose=not args.quiet
        )
        
        if results:
            # Save results
            paths = save_evaluations(results, args.output_dir)
            
            print("\n" + "=" * 60)
            print("EVALUATION COMPLETE")
            print("=" * 60)
            print(f"Evaluated {len(results)} conversations")
            print(f"Results saved to:")
            print(f"  JSON: {paths['json']}")
            print(f"  CSV: {paths['csv']}")
            print(f"  Summary: {paths['summary']}")
            
            # Quick stats
            avg_quality = sum(r.overall_quality_score for r in results) / len(results)
            avg_leak = sum(r.identity_leak_score for r in results) / len(results)
            breakdown_count = sum(1 for r in results if r.breakdown_detected)
            
            print(f"\nQuick Stats:")
            print(f"  Average Quality: {avg_quality:.2f}")
            print(f"  Average Identity Leak: {avg_leak:.2f}")
            print(f"  Breakdowns: {breakdown_count}/{len(results)}")


if __name__ == "__main__":
    main()
