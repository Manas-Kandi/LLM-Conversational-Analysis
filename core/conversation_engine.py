"""
Conversation Engine for managing agent-agent dialogues
Implements the asymmetric information flow architecture
"""
from typing import List, Optional, Callable
from datetime import datetime
from enum import Enum

from core.agent import Agent, AgentFactory
from storage.models import Conversation, Message, AgentRole
from storage.database import Database
from storage.json_storage import JSONStorage
from config import Config


class ConversationStatus(str, Enum):
    """Status of conversation"""
    ACTIVE = "active"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    ERROR = "error"


class ConversationEngine:
    """
    Manages autonomous agent-agent conversation with asymmetric information flow
    
    Architecture:
    - Turn 1: Agent A receives human seed prompt
    - Turn 2+: Agent B responds to Agent A's output (never sees original prompt)
    - Each agent sees accumulated conversation history between agents
    """
    
    def __init__(
        self,
        seed_prompt: str,
        category: str,
        agent_a: Optional[Agent] = None,
        agent_b: Optional[Agent] = None,
        max_turns: int = None,
        context_window_size: int = None,
        database: Optional[Database] = None,
        on_message_callback: Optional[Callable[[Message], None]] = None
    ):
        """
        Initialize conversation engine
        
        Args:
            seed_prompt: Initial human prompt
            category: Research category
            agent_a: Agent A instance (or None to use default)
            agent_b: Agent B instance (or None to use default)
            max_turns: Maximum number of turns (or None for default)
            context_window_size: Number of previous turns to include
            database: Database instance for storage
            on_message_callback: Optional callback function called after each message
        """
        self.seed_prompt = seed_prompt
        self.category = category
        
        # Create agents
        self.agent_a = agent_a or AgentFactory.create_agent_a()
        self.agent_b = agent_b or AgentFactory.create_agent_b()
        
        # Configuration
        self.max_turns = max_turns or Config.DEFAULT_MAX_TURNS
        self.context_window_size = context_window_size or Config.CONTEXT_WINDOW_SIZE
        
        # Storage
        self.database = database or Database(Config.DATABASE_PATH)
        self.json_storage = JSONStorage()  # Auto-save to JSON files
        self.on_message_callback = on_message_callback
        
        # Conversation state
        self.conversation = Conversation(
            seed_prompt=seed_prompt,
            category=category,
            agent_a_model=self.agent_a.model,
            agent_b_model=self.agent_b.model,
            agent_a_temp=self.agent_a.temperature,
            agent_b_temp=self.agent_b.temperature,
            start_time=datetime.now(),
            status=ConversationStatus.ACTIVE.value
        )
        
        # Save initial conversation to database
        self.conversation.id = self.database.create_conversation(self.conversation)
        
        # Flags
        self.is_running = False
        self.should_stop = False
    
    def _get_context_for_agent(self, current_turn: int) -> List[Message]:
        """
        Get conversation context for agent based on context window
        
        For Agent A (turn 1): Includes seed prompt as user message
        For Agent B and subsequent Agent A: Only sees agent-agent conversation history
        """
        if current_turn == 1:
            # First turn: Agent A sees the seed prompt as a user message
            return [Message(
                role=AgentRole.AGENT_B,  # Pretend it's from the "other" agent (user)
                content=self.seed_prompt,
                timestamp=self.conversation.start_time,
                turn_number=0,
                model="human",
                temperature=0.0
            )]
        else:
            # Subsequent turns: Get windowed conversation history
            messages = self.conversation.messages
            if self.context_window_size > 0:
                # Return last N messages
                return messages[-self.context_window_size:]
            else:
                # Return all messages
                return messages
    
    def _add_message(self, message: Message):
        """Add message to conversation and storage"""
        self.conversation.add_message(message)
        self.database.add_message(self.conversation.id, message)
        
        # Call callback if provided
        if self.on_message_callback:
            self.on_message_callback(message)
    
    def run_turn(self) -> Optional[Message]:
        """
        Execute a single conversation turn
        
        Returns:
            Message object or None if conversation should stop
        """
        current_turn = len(self.conversation.messages) + 1
        
        # Check if we've hit max turns
        if current_turn > self.max_turns:
            self._complete_conversation(ConversationStatus.COMPLETED)
            return None
        
        # Check if manually stopped
        if self.should_stop:
            self._complete_conversation(ConversationStatus.TERMINATED)
            return None
        
        # Determine which agent's turn it is
        if current_turn % 2 == 1:
            # Odd turns: Agent A
            agent = self.agent_a
        else:
            # Even turns: Agent B
            agent = self.agent_b
        
        # Get conversation context
        context = self._get_context_for_agent(current_turn)
        
        try:
            # Generate response
            message = agent.generate_response(context)
            
            # Store message
            self._add_message(message)
            
            return message
        
        except Exception as e:
            self._complete_conversation(ConversationStatus.ERROR)
            raise e
    
    def run_conversation(self, blocking: bool = True) -> Conversation:
        """
        Run the complete conversation
        
        Args:
            blocking: If True, runs synchronously. If False, runs in background.
            
        Returns:
            Completed conversation object
        """
        self.is_running = True
        
        while self.is_running and not self.should_stop:
            message = self.run_turn()
            if message is None:
                break
        
        return self.conversation
    
    def stop(self):
        """Stop the conversation gracefully"""
        self.should_stop = True
        self.is_running = False
        self._complete_conversation(ConversationStatus.TERMINATED)
    
    def _complete_conversation(self, status: ConversationStatus):
        """Mark conversation as complete"""
        self.is_running = False
        self.conversation.status = status.value
        self.conversation.end_time = datetime.now()
        
        # Update in database
        self.database.update_conversation_status(
            self.conversation.id,
            status.value,
            self.conversation.end_time
        )
        
        # Auto-save to JSON file
        self.json_storage.save_conversation(self.conversation)
    
    def get_conversation(self) -> Conversation:
        """Get current conversation state"""
        return self.conversation
    
    def get_statistics(self) -> dict:
        """Get conversation statistics"""
        duration = self.conversation.get_duration()
        
        agent_a_messages = self.conversation.get_messages_for_agent(AgentRole.AGENT_A)
        agent_b_messages = self.conversation.get_messages_for_agent(AgentRole.AGENT_B)
        
        total_tokens = sum(
            msg.token_count for msg in self.conversation.messages 
            if msg.token_count
        )
        
        return {
            "total_turns": self.conversation.total_turns,
            "duration_seconds": duration,
            "agent_a_turns": len(agent_a_messages),
            "agent_b_turns": len(agent_b_messages),
            "total_tokens": total_tokens,
            "avg_tokens_per_turn": total_tokens / self.conversation.total_turns if self.conversation.total_turns > 0 else 0,
            "status": self.conversation.status
        }
    
    def __str__(self) -> str:
        return f"Conversation[{self.conversation.id}]: {self.conversation.total_turns} turns, {self.conversation.status}"
