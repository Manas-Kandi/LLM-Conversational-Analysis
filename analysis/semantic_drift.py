"""
Semantic Drift Analysis
Tracks how conversation topic evolves from seed prompt over turns
"""
from typing import Dict, Any, List
import json

from analysis.analyzer import BaseAnalyzer
from storage.models import AnalysisResult


class SemanticDriftAnalyzer(BaseAnalyzer):
    """Analyzes semantic drift from original seed prompt"""
    
    @property
    def analysis_type(self) -> str:
        return "semantic_drift"
    
    def analyze(self) -> AnalysisResult:
        """
        Analyze how conversation topic drifts from seed prompt
        
        Returns:
            AnalysisResult with drift metrics and trajectory
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt()
        
        # Get LLM analysis
        system_prompt = """You are a research analyst studying conversation dynamics.
Your task is to analyze how the conversation topic evolves from the initial seed prompt.
Provide detailed analysis in JSON format."""
        
        response = self._call_llm_for_analysis(prompt, system_prompt)
        
        # Parse response
        try:
            analysis_data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if not valid JSON
            analysis_data = {"raw_analysis": response}
        
        # Calculate drift metrics
        drift_metrics = self._calculate_drift_metrics(analysis_data)
        
        # Generate summary
        summary = self._generate_summary(drift_metrics)
        
        # Create result
        results = {
            "drift_metrics": drift_metrics,
            "llm_analysis": analysis_data,
            "turn_by_turn_relevance": drift_metrics.get("turn_relevance", [])
        }
        
        # Save to database
        analysis_id = self.save_results(results, summary)
        
        return AnalysisResult(
            id=analysis_id,
            conversation_id=self.conversation.id,
            analysis_type=self.analysis_type,
            results=results,
            summary=summary
        )
    
    def _build_analysis_prompt(self) -> str:
        """Build prompt for LLM analysis"""
        conversation_text = self._format_conversation_for_analysis()
        
        prompt = f"""{conversation_text}

ANALYSIS TASK: Semantic Drift Analysis

Analyze how this conversation's topic evolves from the seed prompt. For each turn, assess:
1. Relevance to original seed prompt (0-100%)
2. Main topic/theme being discussed
3. How it connects to or diverges from previous turns
4. Key semantic shifts or topic changes

Provide your analysis in this JSON format:
{{
    "overall_drift_assessment": "high/medium/low with explanation",
    "key_topic_shifts": [
        {{"turn": 5, "from_topic": "...", "to_topic": "...", "trigger": "..."}},
        ...
    ],
    "turn_analysis": [
        {{
            "turn": 1,
            "relevance_to_seed": 95,
            "main_topic": "...",
            "connection": "..."
        }},
        ...
    ],
    "final_topic": "...",
    "drift_trajectory": "description of how conversation evolved"
}}"""
        
        return prompt
    
    def _calculate_drift_metrics(self, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quantitative drift metrics"""
        turn_analysis = llm_analysis.get("turn_analysis", [])
        
        if not turn_analysis:
            return {"error": "No turn analysis available"}
        
        # Extract relevance scores
        relevance_scores = [t.get("relevance_to_seed", 50) for t in turn_analysis]
        
        # Calculate metrics
        initial_relevance = relevance_scores[0] if relevance_scores else 100
        final_relevance = relevance_scores[-1] if relevance_scores else 0
        drift_amount = initial_relevance - final_relevance
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Drift rate (percentage points per turn)
        drift_rate = drift_amount / len(relevance_scores) if relevance_scores else 0
        
        # Classify drift level
        if abs(drift_amount) < 20:
            drift_level = "low"
        elif abs(drift_amount) < 50:
            drift_level = "medium"
        else:
            drift_level = "high"
        
        return {
            "initial_relevance": initial_relevance,
            "final_relevance": final_relevance,
            "drift_amount": drift_amount,
            "drift_rate_per_turn": drift_rate,
            "average_relevance": avg_relevance,
            "drift_level": drift_level,
            "turn_relevance": relevance_scores,
            "total_turns": len(relevance_scores),
            "topic_shifts": llm_analysis.get("key_topic_shifts", []),
            "trajectory_description": llm_analysis.get("drift_trajectory", "")
        }
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        if "error" in metrics:
            return "Unable to complete semantic drift analysis."
        
        summary_parts = []
        
        # Overall drift
        summary_parts.append(
            f"Semantic Drift: {metrics['drift_level'].upper()} "
            f"({metrics['drift_amount']:.1f} percentage points over {metrics['total_turns']} turns)"
        )
        
        # Drift trajectory
        summary_parts.append(
            f"Relevance trajectory: {metrics['initial_relevance']:.0f}% → "
            f"{metrics['final_relevance']:.0f}% "
            f"(avg: {metrics['average_relevance']:.0f}%)"
        )
        
        # Rate
        summary_parts.append(
            f"Drift rate: {metrics['drift_rate_per_turn']:.1f} percentage points per turn"
        )
        
        # Topic shifts
        if metrics.get('topic_shifts'):
            summary_parts.append(
                f"Major topic shifts: {len(metrics['topic_shifts'])}"
            )
        
        return " | ".join(summary_parts)
