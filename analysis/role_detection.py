"""
Role Detection Analysis
Identifies emergent personas and power dynamics between agents
"""
from typing import Dict, Any
import json

from analysis.analyzer import BaseAnalyzer
from storage.models import AnalysisResult


class RoleDetectionAnalyzer(BaseAnalyzer):
    """Analyzes emergent role assignment and persona formation"""
    
    @property
    def analysis_type(self) -> str:
        return "role_detection"
    
    def analyze(self) -> AnalysisResult:
        """
        Analyze emergent roles and power dynamics
        
        Returns:
            AnalysisResult with role assignments and dynamics
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt()
        
        # Get LLM analysis
        system_prompt = """You are a research analyst studying conversational role dynamics.
Your task is to identify emergent personas, power dynamics, and role assignments in AI-to-AI dialogue.
Provide detailed analysis in JSON format."""
        
        response = self._call_llm_for_analysis(prompt, system_prompt)
        
        # Parse response
        try:
            analysis_data = json.loads(response)
        except json.JSONDecodeError:
            analysis_data = {"raw_analysis": response}
        
        # Extract role metrics
        role_metrics = self._extract_role_metrics(analysis_data)
        
        # Generate summary
        summary = self._generate_summary(role_metrics)
        
        # Create result
        results = {
            "role_assignments": analysis_data.get("role_assignments", {}),
            "power_dynamics": analysis_data.get("power_dynamics", {}),
            "role_metrics": role_metrics,
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
        """Build prompt for role analysis"""
        conversation_text = self._format_conversation_for_analysis()
        
        prompt = f"""{conversation_text}

ANALYSIS TASK: Role Detection & Power Dynamics

Analyze the emergent role assignments and interpersonal dynamics. Consider:
1. What persona/role does each agent adopt? (teacher, student, peer, expert, skeptic, etc.)
2. Is there a power dynamic? (dominant, submissive, equal)
3. How stable are these roles across turns?
4. Do agents recognize they're conversing with another AI?
5. What conversational patterns emerge? (questioning, explaining, agreeing, challenging)

Provide your analysis in this JSON format:
{{
    "role_assignments": {{
        "agent_a": {{
            "primary_role": "...",
            "confidence": 0-100,
            "role_descriptors": ["...", "..."],
            "evidence": "key behaviors that support this role"
        }},
        "agent_b": {{
            "primary_role": "...",
            "confidence": 0-100,
            "role_descriptors": ["...", "..."],
            "evidence": "key behaviors that support this role"
        }}
    }},
    "power_dynamics": {{
        "type": "asymmetric-a-dominant/asymmetric-b-dominant/symmetric",
        "description": "...",
        "stability": "stable/fluid/oscillating"
    }},
    "interaction_patterns": [
        "pattern 1",
        "pattern 2"
    ],
    "ai_awareness": {{
        "agent_a_suspects": true/false,
        "agent_b_suspects": true/false,
        "evidence": "..."
    }},
    "role_evolution": "how roles changed over time, if at all"
}}"""
        
        return prompt
    
    def _extract_role_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quantitative metrics from role analysis"""
        role_assignments = analysis.get("role_assignments", {})
        power_dynamics = analysis.get("power_dynamics", {})
        
        # Role confidence scores
        agent_a_confidence = role_assignments.get("agent_a", {}).get("confidence", 0)
        agent_b_confidence = role_assignments.get("agent_b", {}).get("confidence", 0)
        
        # Power dynamic classification
        power_type = power_dynamics.get("type", "unknown")
        power_stability = power_dynamics.get("stability", "unknown")
        
        # AI awareness
        ai_awareness = analysis.get("ai_awareness", {})
        
        return {
            "agent_a_role": role_assignments.get("agent_a", {}).get("primary_role", "unknown"),
            "agent_a_confidence": agent_a_confidence,
            "agent_b_role": role_assignments.get("agent_b", {}).get("primary_role", "unknown"),
            "agent_b_confidence": agent_b_confidence,
            "power_dynamic": power_type,
            "power_stability": power_stability,
            "avg_role_confidence": (agent_a_confidence + agent_b_confidence) / 2,
            "ai_awareness_detected": ai_awareness.get("agent_a_suspects", False) or ai_awareness.get("agent_b_suspects", False),
            "interaction_patterns": analysis.get("interaction_patterns", [])
        }
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        summary_parts = []
        
        # Role assignments
        summary_parts.append(
            f"Agent A: {metrics['agent_a_role'].title()} ({metrics['agent_a_confidence']:.0f}% confidence)"
        )
        summary_parts.append(
            f"Agent B: {metrics['agent_b_role'].title()} ({metrics['agent_b_confidence']:.0f}% confidence)"
        )
        
        # Power dynamic
        summary_parts.append(
            f"Power: {metrics['power_dynamic']} ({metrics['power_stability']})"
        )
        
        # AI awareness
        if metrics['ai_awareness_detected']:
            summary_parts.append("AI awareness: DETECTED")
        
        return " | ".join(summary_parts)
