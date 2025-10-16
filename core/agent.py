"""
Agent wrapper for LLM API interactions
Supports OpenAI and Anthropic models
"""
from typing import List, Dict, Optional
import openai
import anthropic
from datetime import datetime

from config import Config
from storage.models import Message, AgentRole


class Agent:
    """Wrapper for individual LLM agent"""
    
    def __init__(
        self,
        role: AgentRole,
        model: str,
        temperature: float,
        system_prompt: str,
        max_tokens: int = 1000
    ):
        self.role = role
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        
        # Parse model name for custom endpoints (e.g., "nvidia:model-name" or "custom1:model-name")
        self.provider, self.actual_model = self._parse_model_name(model)
        
        # Initialize API clients
        if self.provider == "openai":
            self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        elif self.provider == "anthropic":
            self.anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        elif self.provider in ["nvidia", "custom1", "custom2"]:
            # Get custom endpoint configuration
            api_key, base_url = Config.get_custom_endpoint(self.provider)
            if not api_key or not base_url:
                raise ValueError(f"Missing configuration for {self.provider} endpoint")
            
            # Use OpenAI client with custom base URL
            self.openai_client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
    
    def _parse_model_name(self, model: str) -> tuple[str, str]:
        """
        Parse model name to extract provider and actual model name
        
        Formats:
        - Standard: "gpt-4" -> ("openai", "gpt-4")
        - Custom: "nvidia:qwen/qwen3-next-80b-a3b-instruct" -> ("nvidia", "qwen/qwen3-next-80b-a3b-instruct")
        """
        if ":" in model:
            # Custom endpoint format: "provider:model-name"
            provider, actual_model = model.split(":", 1)
            if provider not in ["nvidia", "custom1", "custom2"]:
                raise ValueError(f"Unknown provider prefix: {provider}. Use nvidia, custom1, or custom2")
            return provider, actual_model
        elif model.startswith("gpt-"):
            return "openai", model
        elif model.startswith("claude-"):
            return "anthropic", model
        else:
            raise ValueError(
                f"Unknown model format: {model}. Use 'gpt-*', 'claude-*', "
                f"or 'provider:model-name' (e.g., 'nvidia:qwen/qwen3-next-80b-a3b-instruct')"
            )
    
    def _build_messages_for_openai(self, conversation_history: List[Message]) -> List[Dict[str, str]]:
        """Build message list for OpenAI API"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for msg in conversation_history:
            # Map agent roles to OpenAI roles
            if msg.role == self.role:
                messages.append({"role": "assistant", "content": msg.content})
            else:
                messages.append({"role": "user", "content": msg.content})
        
        return messages
    
    def _build_messages_for_anthropic(self, conversation_history: List[Message]) -> List[Dict[str, str]]:
        """Build message list for Anthropic API"""
        messages = []
        
        for msg in conversation_history:
            # Map agent roles to Anthropic roles
            if msg.role == self.role:
                messages.append({"role": "assistant", "content": msg.content})
            else:
                messages.append({"role": "user", "content": msg.content})
        
        return messages
    
    def generate_response(self, conversation_history: List[Message]) -> Message:
        """
        Generate a response based on conversation history
        
        Args:
            conversation_history: List of previous messages in conversation
            
        Returns:
            Message object with agent's response
        """
        if self.provider in ["openai", "nvidia", "custom1", "custom2"]:
            # All use OpenAI-compatible API
            return self._generate_openai(conversation_history)
        elif self.provider == "anthropic":
            return self._generate_anthropic(conversation_history)
    
    def _generate_openai(self, conversation_history: List[Message]) -> Message:
        """Generate response using OpenAI API (or OpenAI-compatible APIs)"""
        messages = self._build_messages_for_openai(conversation_history)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.actual_model,  # Use actual model name (without provider prefix)
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                frequency_penalty=0.7,  # Reduce repetition of tokens
                presence_penalty=0.6    # Encourage topic diversity
            )
            
            content = response.choices[0].message.content
            token_count = response.usage.total_tokens
            
            return Message(
                role=self.role,
                content=content,
                timestamp=datetime.now(),
                turn_number=len(conversation_history) + 1,
                model=self.model,
                temperature=self.temperature,
                token_count=token_count,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
        
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _generate_anthropic(self, conversation_history: List[Message]) -> Message:
        """Generate response using Anthropic API"""
        messages = self._build_messages_for_anthropic(conversation_history)
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages
            )
            
            content = response.content[0].text
            token_count = response.usage.input_tokens + response.usage.output_tokens
            
            return Message(
                role=self.role,
                content=content,
                timestamp=datetime.now(),
                turn_number=len(conversation_history) + 1,
                model=self.model,
                temperature=self.temperature,
                token_count=token_count,
                metadata={
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
        
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def __str__(self) -> str:
        return f"{self.role.value}: {self.model} (temp={self.temperature})"


class AgentFactory:
    """Factory for creating configured agents"""
    
    @staticmethod
    def create_agent_a(
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> Agent:
        """Create Agent A with configuration"""
        return Agent(
            role=AgentRole.AGENT_A,
            model=model or Config.AGENT_A_MODEL,
            temperature=temperature or Config.AGENT_A_TEMPERATURE,
            system_prompt=system_prompt or Config.AGENT_A_SYSTEM_PROMPT,
            max_tokens=Config.AGENT_A_MAX_TOKENS
        )
    
    @staticmethod
    def create_agent_b(
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> Agent:
        """Create Agent B with configuration"""
        return Agent(
            role=AgentRole.AGENT_B,
            model=model or Config.AGENT_B_MODEL,
            temperature=temperature or Config.AGENT_B_TEMPERATURE,
            system_prompt=system_prompt or Config.AGENT_B_SYSTEM_PROMPT,
            max_tokens=Config.AGENT_B_MAX_TOKENS
        )
