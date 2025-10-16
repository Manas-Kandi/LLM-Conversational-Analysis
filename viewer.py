#!/usr/bin/env python3
"""
Web-based conversation viewer
Launch a simple web server to browse conversations in a nice table
"""
from flask import Flask, render_template, jsonify, send_from_directory
from pathlib import Path
from storage.database import Database
from config import Config
import json

app = Flask(__name__)
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
def get_conversation(conv_id):
    """Get a specific conversation with messages"""
    conv = db.get_conversation(conv_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    
    messages = db.get_messages(conv_id)
    
    return jsonify({
        "conversation": dict(conv),
        "messages": [dict(msg) for msg in messages]
    })

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
