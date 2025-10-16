"""
Database management for AA Microscope
Uses SQLite for conversation storage and retrieval
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from storage.models import Conversation, Message, AnalysisResult, AgentRole


class Database:
    """SQLite database manager for conversations and analyses"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    seed_prompt TEXT NOT NULL,
                    category TEXT,
                    agent_a_model TEXT NOT NULL,
                    agent_b_model TEXT NOT NULL,
                    agent_a_temp REAL,
                    agent_b_temp REAL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_turns INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL,
                    token_count INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
            
            # Analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    analysis_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    results TEXT NOT NULL,
                    summary TEXT,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_category 
                ON conversations(category)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_status 
                ON conversations(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_conversation 
                ON analysis_results(conversation_id)
            """)
    
    def create_conversation(self, conversation: Conversation) -> int:
        """Create a new conversation and return its ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations 
                (seed_prompt, category, agent_a_model, agent_b_model, 
                 agent_a_temp, agent_b_temp, start_time, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation.seed_prompt,
                conversation.category,
                conversation.agent_a_model,
                conversation.agent_b_model,
                conversation.agent_a_temp,
                conversation.agent_b_temp,
                conversation.start_time.isoformat(),
                conversation.status,
                json.dumps(conversation.metadata)
            ))
            return cursor.lastrowid
    
    def add_message(self, conversation_id: int, message: Message):
        """Add a message to a conversation"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert message
            cursor.execute("""
                INSERT INTO messages 
                (conversation_id, role, content, timestamp, turn_number, 
                 model, temperature, token_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id,
                message.role.value,
                message.content,
                message.timestamp.isoformat(),
                message.turn_number,
                message.model,
                message.temperature,
                message.token_count,
                json.dumps(message.metadata)
            ))
            
            # Update conversation turn count
            cursor.execute("""
                UPDATE conversations 
                SET total_turns = (
                    SELECT COUNT(*) FROM messages WHERE conversation_id = ?
                )
                WHERE id = ?
            """, (conversation_id, conversation_id))
    
    def update_conversation_status(self, conversation_id: int, status: str, end_time: Optional[datetime] = None):
        """Update conversation status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if end_time:
                cursor.execute("""
                    UPDATE conversations 
                    SET status = ?, end_time = ?
                    WHERE id = ?
                """, (status, end_time.isoformat(), conversation_id))
            else:
                cursor.execute("""
                    UPDATE conversations 
                    SET status = ?
                    WHERE id = ?
                """, (status, conversation_id))
    
    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Retrieve a complete conversation with all messages"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get conversation
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            # Build conversation object
            conversation = Conversation(
                id=row["id"],
                seed_prompt=row["seed_prompt"],
                category=row["category"],
                agent_a_model=row["agent_a_model"],
                agent_b_model=row["agent_b_model"],
                agent_a_temp=row["agent_a_temp"],
                agent_b_temp=row["agent_b_temp"],
                start_time=datetime.fromisoformat(row["start_time"]),
                end_time=datetime.fromisoformat(row["end_time"]) if row["end_time"] else None,
                total_turns=row["total_turns"],
                status=row["status"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            
            # Get messages
            cursor.execute("""
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                ORDER BY turn_number
            """, (conversation_id,))
            
            for msg_row in cursor.fetchall():
                message = Message(
                    role=AgentRole(msg_row["role"]),
                    content=msg_row["content"],
                    timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                    turn_number=msg_row["turn_number"],
                    model=msg_row["model"],
                    temperature=msg_row["temperature"],
                    token_count=msg_row["token_count"],
                    metadata=json.loads(msg_row["metadata"]) if msg_row["metadata"] else {}
                )
                conversation.messages.append(message)
            
            return conversation
    
    def list_conversations(self, 
                          category: Optional[str] = None,
                          status: Optional[str] = None,
                          limit: int = 50,
                          offset: int = 0) -> List[Dict[str, Any]]:
        """List conversations with optional filters"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM conversations WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    "id": row["id"],
                    "seed_prompt": row["seed_prompt"][:100] + "..." if len(row["seed_prompt"]) > 100 else row["seed_prompt"],
                    "category": row["category"],
                    "agent_a_model": row["agent_a_model"],
                    "agent_b_model": row["agent_b_model"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "total_turns": row["total_turns"],
                    "status": row["status"]
                })
            
            return conversations
    
    def save_analysis(self, analysis: AnalysisResult) -> int:
        """Save analysis results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_results 
                (conversation_id, analysis_type, timestamp, results, summary, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                analysis.conversation_id,
                analysis.analysis_type,
                analysis.timestamp.isoformat(),
                json.dumps(analysis.results),
                analysis.summary,
                json.dumps(analysis.metadata)
            ))
            return cursor.lastrowid
    
    def get_analyses(self, conversation_id: int) -> List[AnalysisResult]:
        """Get all analyses for a conversation"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM analysis_results 
                WHERE conversation_id = ? 
                ORDER BY timestamp DESC
            """, (conversation_id,))
            
            analyses = []
            for row in cursor.fetchall():
                analysis = AnalysisResult(
                    id=row["id"],
                    conversation_id=row["conversation_id"],
                    analysis_type=row["analysis_type"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    results=json.loads(row["results"]),
                    summary=row["summary"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                )
                analyses.append(analysis)
            
            return analyses
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total conversations
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            total_conversations = cursor.fetchone()["count"]
            
            # Conversations by status
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM conversations 
                GROUP BY status
            """)
            by_status = {row["status"]: row["count"] for row in cursor.fetchall()}
            
            # Conversations by category
            cursor.execute("""
                SELECT category, COUNT(*) as count 
                FROM conversations 
                GROUP BY category
            """)
            by_category = {row["category"]: row["count"] for row in cursor.fetchall()}
            
            # Total messages
            cursor.execute("SELECT COUNT(*) as count FROM messages")
            total_messages = cursor.fetchone()["count"]
            
            # Total analyses
            cursor.execute("SELECT COUNT(*) as count FROM analysis_results")
            total_analyses = cursor.fetchone()["count"]
            
            return {
                "total_conversations": total_conversations,
                "by_status": by_status,
                "by_category": by_category,
                "total_messages": total_messages,
                "total_analyses": total_analyses
            }
