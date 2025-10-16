"""
Base analyzer framework for conversation analysis
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import openai
import anthropic

from storage.models import Conversation, AnalysisResult
from storage.database import Database
from config import Config


class BaseAnalyzer(ABC):
    """Base class for conversation analyzers"""
    
    def __init__(self, conversation: Conversation, database: Optional[Database] = None):
        self.conversation = conversation
        self.database = database or Database(Config.DATABASE_PATH)
        self.analysis_model = Config.ANALYSIS_MODEL
        
        # Initialize LLM client for analysis
        if self.analysis_model.startswith("gpt-"):
            self.llm_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            self.llm_provider = "openai"
        elif self.analysis_model.startswith("claude-"):
            self.llm_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            self.llm_provider = "anthropic"
    
    @abstractmethod
    def analyze(self) -> AnalysisResult:
        """
        Perform analysis on conversation
        Must be implemented by subclasses
        """
        pass
    
    @property
    @abstractmethod
    def analysis_type(self) -> str:
        """Return the type of analysis"""
        pass
    
    def _call_llm_for_analysis(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call LLM to help with analysis
        
        Args:
            prompt: Analysis prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLM response as string
        """
        if self.llm_provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.llm_client.chat.completions.create(
                model=self.analysis_model,
                messages=messages,
                temperature=0.3  # Lower temperature for more focused analysis
            )
            return response.choices[0].message.content
        
        elif self.llm_provider == "anthropic":
            response = self.llm_client.messages.create(
                model=self.analysis_model,
                max_tokens=2000,
                temperature=0.3,
                system=system_prompt or "You are a helpful research analyst studying AI conversation patterns.",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    def _format_conversation_for_analysis(self) -> str:
        """Format conversation as readable text for analysis"""
        lines = []
        lines.append(f"=== CONVERSATION ANALYSIS ===")
        lines.append(f"Seed Prompt: {self.conversation.seed_prompt}")
        lines.append(f"Category: {self.conversation.category}")
        lines.append(f"Agent A Model: {self.conversation.agent_a_model}")
        lines.append(f"Agent B Model: {self.conversation.agent_b_model}")
        lines.append(f"Total Turns: {self.conversation.total_turns}")
        lines.append("")
        lines.append("=== CONVERSATION ===")
        
        for msg in self.conversation.messages:
            agent_label = "Agent A" if msg.role.value == "agent_a" else "Agent B"
            lines.append(f"\n[Turn {msg.turn_number}] {agent_label}:")
            lines.append(msg.content)
        
        return "\n".join(lines)
    
    def save_results(self, results: Dict[str, Any], summary: str) -> int:
        """Save analysis results to database"""
        analysis = AnalysisResult(
            conversation_id=self.conversation.id,
            analysis_type=self.analysis_type,
            timestamp=datetime.now(),
            results=results,
            summary=summary
        )
        return self.database.save_analysis(analysis)
    
    def get_conversation_text_by_agent(self) -> Dict[str, str]:
        """Get separate text for each agent's messages"""
        from storage.models import AgentRole
        
        agent_a_msgs = [
            f"Turn {msg.turn_number}: {msg.content}"
            for msg in self.conversation.messages
            if msg.role == AgentRole.AGENT_A
        ]
        
        agent_b_msgs = [
            f"Turn {msg.turn_number}: {msg.content}"
            for msg in self.conversation.messages
            if msg.role == AgentRole.AGENT_B
        ]
        
        return {
            "agent_a": "\n\n".join(agent_a_msgs),
            "agent_b": "\n\n".join(agent_b_msgs)
        }
    
    def calculate_basic_stats(self) -> Dict[str, Any]:
        """Calculate basic statistical metrics"""
        messages = self.conversation.messages
        
        if not messages:
            return {}
        
        # Message lengths
        lengths = [len(msg.content) for msg in messages]
        
        # Token counts
        token_counts = [msg.token_count for msg in messages if msg.token_count]
        
        # Calculate stats
        return {
            "total_messages": len(messages),
            "avg_message_length": sum(lengths) / len(lengths),
            "min_message_length": min(lengths),
            "max_message_length": max(lengths),
            "total_characters": sum(lengths),
            "avg_tokens_per_message": sum(token_counts) / len(token_counts) if token_counts else 0,
            "total_tokens": sum(token_counts)
        }
