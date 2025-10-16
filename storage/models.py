"""
Data models for AA Microscope storage
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import json


class AgentRole(str, Enum):
    """Agent role in conversation"""
    AGENT_A = "agent_a"
    AGENT_B = "agent_b"


@dataclass
class Message:
    """A single message in the conversation"""
    role: AgentRole
    content: str
    timestamp: datetime
    turn_number: int
    model: str
    temperature: float
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "turn_number": self.turn_number,
            "model": self.model,
            "temperature": self.temperature,
            "token_count": self.token_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary"""
        return cls(
            role=AgentRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            turn_number=data["turn_number"],
            model=data["model"],
            temperature=data["temperature"],
            token_count=data.get("token_count"),
            metadata=data.get("metadata", {})
        )


@dataclass
class Conversation:
    """A complete agent-agent conversation"""
    id: Optional[int] = None
    seed_prompt: str = ""
    category: str = ""
    agent_a_model: str = ""
    agent_b_model: str = ""
    agent_a_temp: float = 0.7
    agent_b_temp: float = 0.7
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_turns: int = 0
    status: str = "active"  # active, completed, terminated
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.total_turns = len(self.messages)
    
    def get_duration(self) -> Optional[float]:
        """Get conversation duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def get_messages_for_agent(self, role: AgentRole) -> List[Message]:
        """Get all messages from a specific agent"""
        return [m for m in self.messages if m.role == role]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "seed_prompt": self.seed_prompt,
            "category": self.category,
            "agent_a_model": self.agent_a_model,
            "agent_b_model": self.agent_b_model,
            "agent_a_temp": self.agent_a_temp,
            "agent_b_temp": self.agent_b_temp,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_turns": self.total_turns,
            "status": self.status,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create from dictionary"""
        conv = cls(
            id=data.get("id"),
            seed_prompt=data["seed_prompt"],
            category=data["category"],
            agent_a_model=data["agent_a_model"],
            agent_b_model=data["agent_b_model"],
            agent_a_temp=data["agent_a_temp"],
            agent_b_temp=data["agent_b_temp"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            total_turns=data["total_turns"],
            status=data["status"],
            metadata=data.get("metadata", {})
        )
        conv.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return conv


@dataclass
class AnalysisResult:
    """Result of conversation analysis"""
    id: Optional[int] = None
    conversation_id: int = 0
    analysis_type: str = ""  # semantic_drift, role_detection, pattern_recognition, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    results: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "analysis_type": self.analysis_type,
            "timestamp": self.timestamp.isoformat(),
            "results": self.results,
            "summary": self.summary,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create from dictionary"""
        return cls(
            id=data.get("id"),
            conversation_id=data["conversation_id"],
            analysis_type=data["analysis_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            results=data["results"],
            summary=data["summary"],
            metadata=data.get("metadata", {})
        )
