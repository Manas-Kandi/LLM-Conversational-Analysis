"""
Simple JSON file storage for conversations
Each conversation is saved as a separate JSON file
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from storage.models import Conversation, Message


class JSONStorage:
    """Store conversations as individual JSON files"""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path("conversations_json")
        self.storage_dir.mkdir(exist_ok=True)
    
    def save_conversation(self, conversation: Conversation) -> Path:
        """Save a conversation to a JSON file"""
        # Create filename with timestamp and ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conv_{conversation.id}_{timestamp}.json"
        filepath = self.storage_dir / filename
        
        # Convert conversation to dict
        data = {
            "id": conversation.id,
            "metadata": {
                "seed_prompt": conversation.seed_prompt,
                "category": conversation.category,
                "status": conversation.status,
                "start_time": conversation.start_time.isoformat() if conversation.start_time else None,
                "end_time": conversation.end_time.isoformat() if conversation.end_time else None,
                "total_turns": conversation.total_turns,
            },
            "agents": {
                "agent_a": {
                    "model": conversation.agent_a_model,
                    "temperature": conversation.agent_a_temp,
                },
                "agent_b": {
                    "model": conversation.agent_b_model,
                    "temperature": conversation.agent_b_temp,
                }
            },
            "messages": [
                {
                    "turn": msg.turn_number,
                    "role": msg.role.value,
                    "content": msg.content,
                    "model": msg.model,
                    "temperature": msg.temperature,
                    "token_count": msg.token_count,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "metadata": msg.metadata,
                }
                for msg in conversation.messages
            ],
            "statistics": {
                "total_messages": len(conversation.messages),
                "total_tokens": sum(msg.token_count or 0 for msg in conversation.messages),
                "average_tokens_per_message": (
                    sum(msg.token_count or 0 for msg in conversation.messages) // len(conversation.messages)
                    if conversation.messages else 0
                ),
                "duration_seconds": conversation.get_duration() if conversation.end_time else None,
            }
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved: {filepath}")
        return filepath
    
    def load_conversation(self, filepath: Path) -> Dict[str, Any]:
        """Load a conversation from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_conversations(self) -> List[Path]:
        """List all conversation JSON files"""
        return sorted(self.storage_dir.glob("conv_*.json"), reverse=True)
    
    def get_latest(self, n: int = 10) -> List[Path]:
        """Get the n most recent conversations"""
        return self.list_conversations()[:n]
