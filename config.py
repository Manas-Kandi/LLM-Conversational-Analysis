"""
Configuration management for AA Microscope
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
STORAGE_DIR = PROJECT_ROOT / "storage"
EXPORTS_DIR = PROJECT_ROOT / "exports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
STORAGE_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class Config:
    """Central configuration class"""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # NVIDIA NIM API
    NVIDIA_API_KEY: Optional[str] = os.getenv("NVIDIA_API_KEY")
    NVIDIA_BASE_URL: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    # Custom OpenAI-compatible endpoints
    CUSTOM_API_KEY_1: Optional[str] = os.getenv("CUSTOM_API_KEY_1")
    CUSTOM_BASE_URL_1: Optional[str] = os.getenv("CUSTOM_BASE_URL_1")
    CUSTOM_API_KEY_2: Optional[str] = os.getenv("CUSTOM_API_KEY_2")
    CUSTOM_BASE_URL_2: Optional[str] = os.getenv("CUSTOM_BASE_URL_2")
    
    # Agent A Configuration
    AGENT_A_MODEL: str = os.getenv("AGENT_A_MODEL", "nvidia:meta/llama-3.1-70b-instruct")
    AGENT_A_TEMPERATURE: float = float(os.getenv("AGENT_A_TEMPERATURE", "0.7"))
    AGENT_A_MAX_TOKENS: int = int(os.getenv("AGENT_A_MAX_TOKENS", "1000"))
    
    # Agent B Configuration
    AGENT_B_MODEL: str = os.getenv("AGENT_B_MODEL", "nvidia:meta/llama-3.1-70b-instruct")
    AGENT_B_TEMPERATURE: float = float(os.getenv("AGENT_B_TEMPERATURE", "0.7"))
    AGENT_B_MAX_TOKENS: int = int(os.getenv("AGENT_B_MAX_TOKENS", "1000"))
    
    # Conversation Settings
    DEFAULT_MAX_TURNS: int = int(os.getenv("DEFAULT_MAX_TURNS", "30"))
    CONTEXT_WINDOW_SIZE: int = int(os.getenv("CONTEXT_WINDOW_SIZE", "10"))
    
    # Database
    DATABASE_PATH: Path = STORAGE_DIR / "conversations.db"
    
    # Analysis
    ANALYSIS_MODEL: str = os.getenv("ANALYSIS_MODEL", "nvidia:meta/llama-3.1-70b-instruct")
    ENABLE_AUTO_ANALYSIS: bool = os.getenv("ENABLE_AUTO_ANALYSIS", "true").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = LOGS_DIR / "aa_microscope.log"
    
    # System prompts - Differentiated to reduce repetition
    AGENT_A_SYSTEM_PROMPT: str = """You are having a natural conversation. Respond directly and authentically.
Be concise and avoid over-explaining. Don't be overly polite or repetitive.
Share your genuine thoughts and reactions."""
    
    AGENT_B_SYSTEM_PROMPT: str = """You are engaged in a conversation. Respond naturally and build on what was said.
Be direct and authentic. Avoid echoing the other person's phrasing.
Focus on adding new perspectives rather than summarizing."""
    
    @classmethod
    def get_custom_endpoint(cls, provider: str) -> tuple[Optional[str], Optional[str]]:
        """Get custom endpoint configuration for a provider"""
        if provider == "nvidia":
            return cls.NVIDIA_API_KEY, cls.NVIDIA_BASE_URL
        elif provider == "custom1":
            return cls.CUSTOM_API_KEY_1, cls.CUSTOM_BASE_URL_1
        elif provider == "custom2":
            return cls.CUSTOM_API_KEY_2, cls.CUSTOM_BASE_URL_2
        return None, None
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that essential configuration is present"""
        errors = []
        
        # Check if at least one API key is configured
        has_api_key = any([
            cls.OPENAI_API_KEY,
            cls.ANTHROPIC_API_KEY,
            cls.NVIDIA_API_KEY,
            cls.CUSTOM_API_KEY_1,
            cls.CUSTOM_API_KEY_2
        ])
        
        if not has_api_key:
            errors.append("At least one API key must be set (OpenAI, Anthropic, NVIDIA, or Custom)")
        
        # Validate agent A model
        if cls.AGENT_A_MODEL.startswith("gpt-") and not cls.OPENAI_API_KEY:
            errors.append("OpenAI API key required for GPT models")
        elif cls.AGENT_A_MODEL.startswith("claude-") and not cls.ANTHROPIC_API_KEY:
            errors.append("Anthropic API key required for Claude models")
        elif cls.AGENT_A_MODEL.startswith("nvidia:") and not cls.NVIDIA_API_KEY:
            errors.append("NVIDIA API key required for NVIDIA NIM models")
        elif cls.AGENT_A_MODEL.startswith("custom1:") and not cls.CUSTOM_API_KEY_1:
            errors.append("CUSTOM_API_KEY_1 required for custom1: models")
        elif cls.AGENT_A_MODEL.startswith("custom2:") and not cls.CUSTOM_API_KEY_2:
            errors.append("CUSTOM_API_KEY_2 required for custom2: models")
        
        # Validate agent B model
        if cls.AGENT_B_MODEL.startswith("gpt-") and not cls.OPENAI_API_KEY:
            errors.append("OpenAI API key required for GPT models (Agent B)")
        elif cls.AGENT_B_MODEL.startswith("claude-") and not cls.ANTHROPIC_API_KEY:
            errors.append("Anthropic API key required for Claude models (Agent B)")
        elif cls.AGENT_B_MODEL.startswith("nvidia:") and not cls.NVIDIA_API_KEY:
            errors.append("NVIDIA API key required for NVIDIA NIM models (Agent B)")
        elif cls.AGENT_B_MODEL.startswith("custom1:") and not cls.CUSTOM_API_KEY_1:
            errors.append("CUSTOM_API_KEY_1 required for custom1: models (Agent B)")
        elif cls.AGENT_B_MODEL.startswith("custom2:") and not cls.CUSTOM_API_KEY_2:
            errors.append("CUSTOM_API_KEY_2 required for custom2: models (Agent B)")
        
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def display(cls):
        """Display current configuration"""
        print("🔬 AA Microscope Configuration")
        print(f"Agent A: {cls.AGENT_A_MODEL} (temp={cls.AGENT_A_TEMPERATURE})")
        print(f"Agent B: {cls.AGENT_B_MODEL} (temp={cls.AGENT_B_TEMPERATURE})")
        print(f"Max Turns: {cls.DEFAULT_MAX_TURNS}")
        print(f"Context Window: {cls.CONTEXT_WINDOW_SIZE} turns")
        print(f"Database: {cls.DATABASE_PATH}")


# Create singleton instance
config = Config()
