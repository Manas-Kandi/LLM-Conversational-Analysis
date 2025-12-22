#!/usr/bin/env python3
"""
Web-based conversation viewer
Launch a simple web server to browse conversations in a nice table
"""
from flask import Flask, render_template, jsonify
from pathlib import Path
from storage.database import Database
from config import Config
import json

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
db = Database(Config.DATABASE_PATH)

@app.route('/')
def index():
    """Main page with conversation table"""
    return render_template('viewer.html')

@app.route('/api/conversations')
def get_conversations():
    """Get all conversations as JSON"""
    conversations = db.list_conversations(limit=1000)
    return jsonify(conversations)

@app.route('/api/conversation/<int:conv_id>')
def get_conversation_detail(conv_id):
    """Get a specific conversation with messages"""
    conv = db.get_conversation(conv_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    
    # Serialize conversation metadata
    conv_data = {
        "id": conv.id,
        "seed_prompt": conv.seed_prompt,
        "category": conv.category,
        "agent_a_model": conv.agent_a_model,
        "agent_b_model": conv.agent_b_model,
        "agent_a_temp": conv.agent_a_temp,
        "agent_b_temp": conv.agent_b_temp,
        "start_time": conv.start_time.isoformat() if conv.start_time else None,
        "end_time": conv.end_time.isoformat() if conv.end_time else None,
        "total_turns": conv.total_turns,
        "status": conv.status
    }
    
    # Serialize messages
    messages_data = [{
        "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
        "content": msg.content,
        "turn_number": msg.turn_number,
        "model": msg.model,
        "temperature": msg.temperature,
        "token_count": msg.token_count
    } for msg in conv.messages]
    
    return jsonify({
        "conversation": conv_data,
        "messages": messages_data
    })

@app.route('/api/conversation/create', methods=['POST'])
def create_conversation():
    """Create a new conversation from JSON data"""
    from flask import request
    from datetime import datetime
    from storage.models import Conversation, Message, AgentRole
    
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        # Create conversation object
        conv = Conversation(
            seed_prompt=data.get('seed_prompt', ''),
            category=data.get('category', 'manual_entry'),
            agent_a_model=data.get('agent_a_model', 'unknown'),
            agent_b_model=data.get('agent_b_model', 'unknown'),
            agent_a_temp=data.get('agent_a_temp', 0.7),
            agent_b_temp=data.get('agent_b_temp', 0.7),
            start_time=datetime.now(),
            status=data.get('status', 'completed'),
            metadata=data.get('metadata', {})
        )
        
        # Save conversation to get ID
        conv_id = db.create_conversation(conv)
        
        # Add messages if provided
        messages = data.get('messages', [])
        for i, msg_data in enumerate(messages):
            msg = Message(
                role=AgentRole(msg_data['role']),
                content=msg_data['content'],
                timestamp=datetime.now(),
                turn_number=msg_data.get('turn_number', i + 1),
                model=msg_data.get('model', 'unknown'),
                temperature=msg_data.get('temperature', 0.7),
                token_count=msg_data.get('token_count', 0),
                metadata=msg_data.get('metadata', {})
            )
            db.add_message(conv_id, msg)
            
        return jsonify({"success": True, "id": conv_id}), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get database statistics"""
    stats = db.get_statistics()
    return jsonify(stats)

def main():
    print("🔬 AA Microscope - Conversation Viewer")
    print("=" * 50)
    print("\n✨ Starting web server...")
    print("\n🌐 Open in your browser:")
    print("   👉 http://localhost:5000")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, port=5000)

if __name__ == '__main__':
    main()
