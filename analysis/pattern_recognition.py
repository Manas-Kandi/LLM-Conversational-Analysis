"""
Pattern Recognition Analysis
Identifies recurring themes, conversational patterns, and emergent phenomena
"""
from typing import Dict, Any, List
import json

from analysis.analyzer import BaseAnalyzer
from storage.models import AnalysisResult


class PatternRecognitionAnalyzer(BaseAnalyzer):
    """Analyzes recurring patterns and emergent phenomena"""
    
    @property
    def analysis_type(self) -> str:
        return "pattern_recognition"
    
    def analyze(self) -> AnalysisResult:
        """
        Identify patterns and emergent phenomena in conversation
        
        Returns:
            AnalysisResult with identified patterns
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt()
        
        # Get LLM analysis
        system_prompt = """You are a research analyst studying emergent patterns in AI dialogue.
Your task is to identify recurring themes, conversational patterns, and unexpected phenomena.
Look for creativity, circular reasoning, persuasion tactics, emotional arcs, and meta-cognitive moments.
Provide detailed analysis in JSON format."""
        
        response = self._call_llm_for_analysis(prompt, system_prompt)
        
        # Parse response
        try:
            analysis_data = json.loads(response)
        except json.JSONDecodeError:
            analysis_data = {"raw_analysis": response}
        
        # Extract pattern metrics
        pattern_metrics = self._extract_pattern_metrics(analysis_data)
        
        # Generate summary
        summary = self._generate_summary(pattern_metrics, analysis_data)
        
        # Create result
        results = {
            "patterns": analysis_data.get("patterns", []),
            "phenomena": analysis_data.get("phenomena", []),
            "pattern_metrics": pattern_metrics,
            "notable_moments": analysis_data.get("notable_moments", []),
            "llm_analysis": analysis_data
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
        """Build prompt for pattern analysis"""
        conversation_text = self._format_conversation_for_analysis()
        
        prompt = f"""{conversation_text}

ANALYSIS TASK: Pattern Recognition & Emergent Phenomena

Analyze this conversation for:

1. **Recurring Patterns**: What conversational structures repeat?
   - Question-answer cycles
   - Agreement-building sequences
   - Topic introduction patterns
   - Rhetorical strategies

2. **Emergent Phenomena**: What unexpected behaviors emerge?
   - Creativity vs. recycling of ideas
   - Novel insights or platitudes
   - Conversational breakdown points
   - Meta-cognitive moments (awareness of the conversation itself)
   - Humor, irony, or wordplay

3. **Information Dynamics**:
   - Are they generating new ideas or recombining existing knowledge?
   - Evidence of "persuasion" or influence
   - Conflict, agreement, or synthesis patterns

4. **Notable Moments**: Specific turns that are particularly interesting

Provide your analysis in this JSON format:
{{
    "patterns": [
        {{
            "pattern_name": "...",
            "description": "...",
            "frequency": "high/medium/low",
            "examples": ["turn X", "turn Y"]
        }},
        ...
    ],
    "phenomena": [
        {{
            "phenomenon": "...",
            "description": "...",
            "significance": "why this is interesting",
            "examples": ["turn X"]
        }},
        ...
    ],
    "creativity_assessment": {{
        "level": "high/medium/low",
        "evidence": "...",
        "novel_vs_recycled": "..."
    }},
    "information_dynamics": {{
        "type": "generative/recycling/mixed",
        "description": "..."
    }},
    "conversational_health": {{
        "status": "thriving/stable/degrading/collapsed",
        "evidence": "..."
    }},
    "notable_moments": [
        {{
            "turn": X,
            "description": "what makes this turn interesting",
            "category": "breakthrough/breakdown/meta-cognitive/humorous/other"
        }},
        ...
    ]
}}"""
        
        return prompt
    
    def _extract_pattern_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quantitative metrics from pattern analysis"""
        patterns = analysis.get("patterns", [])
        phenomena = analysis.get("phenomena", [])
        notable_moments = analysis.get("notable_moments", [])
        
        creativity = analysis.get("creativity_assessment", {})
        info_dynamics = analysis.get("information_dynamics", {})
        health = analysis.get("conversational_health", {})
        
        return {
            "total_patterns": len(patterns),
            "total_phenomena": len(phenomena),
            "total_notable_moments": len(notable_moments),
            "creativity_level": creativity.get("level", "unknown"),
            "information_type": info_dynamics.get("type", "unknown"),
            "conversational_health": health.get("status", "unknown"),
            "pattern_categories": [p.get("pattern_name") for p in patterns],
            "phenomenon_types": [ph.get("phenomenon") for ph in phenomena]
        }
    
    def _generate_summary(self, metrics: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        summary_parts = []
        
        # Patterns found
        summary_parts.append(
            f"Patterns: {metrics['total_patterns']} identified"
        )
        
        # Phenomena
        summary_parts.append(
            f"Emergent phenomena: {metrics['total_phenomena']}"
        )
        
        # Creativity
        summary_parts.append(
            f"Creativity: {metrics['creativity_level']}"
        )
        
        # Information dynamics
        summary_parts.append(
            f"Info dynamics: {metrics['information_type']}"
        )
        
        # Health
        summary_parts.append(
            f"Health: {metrics['conversational_health']}"
        )
        
        # Notable moments
        if metrics['total_notable_moments'] > 0:
            summary_parts.append(
                f"Notable moments: {metrics['total_notable_moments']}"
            )
        
        return " | ".join(summary_parts)
