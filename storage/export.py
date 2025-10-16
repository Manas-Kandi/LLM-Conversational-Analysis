"""
Export conversations to friendly, readable formats
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from storage.models import Conversation, Message, AgentRole
from storage.database import Database


class ConversationExporter:
    """Export conversations in various friendly formats"""
    
    def __init__(self, database: Database):
        self.database = database
    
    def export_to_markdown(self, conversation_id: int, output_path: Optional[Path] = None) -> str:
        """Export conversation to a beautiful Markdown file"""
        conv = self.database.get_conversation(conversation_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        messages = self.database.get_messages(conversation_id)
        
        # Build markdown content
        md_lines = []
        md_lines.append(f"# 🔬 AA Microscope Conversation #{conversation_id}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # Metadata section
        md_lines.append("## 📋 Conversation Metadata")
        md_lines.append("")
        md_lines.append(f"- **Category**: {conv['category']}")
        md_lines.append(f"- **Status**: {conv['status']}")
        md_lines.append(f"- **Started**: {conv['start_time']}")
        if conv['end_time']:
            md_lines.append(f"- **Ended**: {conv['end_time']}")
        md_lines.append(f"- **Total Turns**: {conv['total_turns']}")
        md_lines.append("")
        
        # Models section
        md_lines.append("## 🤖 Agent Configuration")
        md_lines.append("")
        md_lines.append(f"**Agent A**: `{conv['agent_a_model']}` (temp: {conv['agent_a_temp']})")
        md_lines.append("")
        md_lines.append(f"**Agent B**: `{conv['agent_b_model']}` (temp: {conv['agent_b_temp']})")
        md_lines.append("")
        
        # Seed prompt
        md_lines.append("## 🌱 Seed Prompt")
        md_lines.append("")
        md_lines.append(f"> {conv['seed_prompt']}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # Conversation
        md_lines.append("## 💬 Conversation")
        md_lines.append("")
        
        for msg in messages:
            agent_name = "🔵 Agent A" if msg['role'] == 'agent_a' else "🟣 Agent B"
            md_lines.append(f"### {agent_name} - Turn {msg['turn_number']}")
            md_lines.append("")
            md_lines.append(msg['content'])
            md_lines.append("")
            md_lines.append(f"*Model: {msg['model']} | Tokens: {msg['token_count'] or 'N/A'} | Time: {msg['timestamp']}*")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
        
        # Statistics
        md_lines.append("## 📊 Statistics")
        md_lines.append("")
        total_tokens = sum(msg['token_count'] or 0 for msg in messages)
        md_lines.append(f"- **Total Messages**: {len(messages)}")
        md_lines.append(f"- **Total Tokens**: {total_tokens}")
        md_lines.append(f"- **Average Tokens per Message**: {total_tokens // len(messages) if messages else 0}")
        md_lines.append("")
        
        markdown_content = "\n".join(md_lines)
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown_content)
        
        return markdown_content
    
    def export_to_html(self, conversation_id: int, output_path: Optional[Path] = None) -> str:
        """Export conversation to a beautiful HTML file"""
        conv = self.database.get_conversation(conversation_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        messages = self.database.get_messages(conversation_id)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AA Microscope - Conversation #{conversation_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .metadata {{
            background: #f8f9fa;
            padding: 30px;
            border-bottom: 3px solid #667eea;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .metadata-item {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        
        .metadata-item strong {{
            color: #667eea;
            display: block;
            margin-bottom: 5px;
        }}
        
        .seed-prompt {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            margin: 30px;
            border-radius: 10px;
            font-style: italic;
        }}
        
        .conversation {{
            padding: 30px;
        }}
        
        .message {{
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 15px;
            position: relative;
            animation: fadeIn 0.5s ease-in;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .message.agent-a {{
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 5px solid #2196F3;
        }}
        
        .message.agent-b {{
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            border-left: 5px solid #9c27b0;
        }}
        
        .message-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(0,0,0,0.1);
        }}
        
        .agent-name {{
            font-weight: bold;
            font-size: 1.2em;
        }}
        
        .agent-a .agent-name {{
            color: #1976D2;
        }}
        
        .agent-b .agent-name {{
            color: #7B1FA2;
        }}
        
        .turn-number {{
            background: rgba(0,0,0,0.1);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        
        .message-content {{
            font-size: 1.05em;
            line-height: 1.8;
            color: #2c3e50;
        }}
        
        .message-meta {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(0,0,0,0.1);
            font-size: 0.85em;
            color: #666;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            border-top: 3px solid #667eea;
        }}
        
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 AA Microscope</h1>
            <p>Conversation #{conversation_id}</p>
        </div>
        
        <div class="metadata">
            <h2>📋 Conversation Details</h2>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>Category</strong>
                    {conv['category']}
                </div>
                <div class="metadata-item">
                    <strong>Status</strong>
                    {conv['status']}
                </div>
                <div class="metadata-item">
                    <strong>Started</strong>
                    {conv['start_time']}
                </div>
                <div class="metadata-item">
                    <strong>Total Turns</strong>
                    {conv['total_turns']}
                </div>
            </div>
            
            <h3 style="margin-top: 30px; color: #667eea;">🤖 Agent Configuration</h3>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>Agent A</strong>
                    {conv['agent_a_model']}<br>
                    <small>Temperature: {conv['agent_a_temp']}</small>
                </div>
                <div class="metadata-item">
                    <strong>Agent B</strong>
                    {conv['agent_b_model']}<br>
                    <small>Temperature: {conv['agent_b_temp']}</small>
                </div>
            </div>
        </div>
        
        <div class="seed-prompt">
            <strong>🌱 Seed Prompt:</strong><br>
            {conv['seed_prompt']}
        </div>
        
        <div class="conversation">
            <h2 style="margin-bottom: 30px; color: #667eea;">💬 Conversation</h2>
"""
        
        for msg in messages:
            agent_class = "agent-a" if msg['role'] == 'agent_a' else "agent-b"
            agent_name = "🔵 Agent A" if msg['role'] == 'agent_a' else "🟣 Agent B"
            
            html += f"""
            <div class="message {agent_class}">
                <div class="message-header">
                    <span class="agent-name">{agent_name}</span>
                    <span class="turn-number">Turn {msg['turn_number']}</span>
                </div>
                <div class="message-content">
                    {msg['content'].replace(chr(10), '<br>')}
                </div>
                <div class="message-meta">
                    <div class="meta-item">
                        <strong>Model:</strong> {msg['model']}
                    </div>
                    <div class="meta-item">
                        <strong>Tokens:</strong> {msg['token_count'] or 'N/A'}
                    </div>
                    <div class="meta-item">
                        <strong>Time:</strong> {msg['timestamp']}
                    </div>
                </div>
            </div>
"""
        
        total_tokens = sum(msg['token_count'] or 0 for msg in messages)
        
        html += f"""
        </div>
        
        <div class="footer">
            <h2>📊 Statistics</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{len(messages)}</div>
                    <div class="stat-label">Total Messages</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{total_tokens}</div>
                    <div class="stat-label">Total Tokens</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{total_tokens // len(messages) if messages else 0}</div>
                    <div class="stat-label">Avg Tokens/Message</div>
                </div>
            </div>
            <p style="margin-top: 30px; color: #666;">
                Generated by AA Microscope on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html)
        
        return html
    
    def export_to_json(self, conversation_id: int, output_path: Optional[Path] = None) -> str:
        """Export conversation to structured JSON"""
        conv = self.database.get_conversation(conversation_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        messages = self.database.get_messages(conversation_id)
        
        data = {
            "conversation_id": conversation_id,
            "metadata": {
                "seed_prompt": conv['seed_prompt'],
                "category": conv['category'],
                "status": conv['status'],
                "start_time": conv['start_time'],
                "end_time": conv['end_time'],
                "total_turns": conv['total_turns']
            },
            "agents": {
                "agent_a": {
                    "model": conv['agent_a_model'],
                    "temperature": conv['agent_a_temp'],
                    "max_tokens": conv['agent_a_max_tokens']
                },
                "agent_b": {
                    "model": conv['agent_b_model'],
                    "temperature": conv['agent_b_temp'],
                    "max_tokens": conv['agent_b_max_tokens']
                }
            },
            "messages": [
                {
                    "turn": msg['turn_number'],
                    "role": msg['role'],
                    "content": msg['content'],
                    "model": msg['model'],
                    "temperature": msg['temperature'],
                    "token_count": msg['token_count'],
                    "timestamp": msg['timestamp'],
                    "metadata": json.loads(msg['metadata']) if msg['metadata'] else {}
                }
                for msg in messages
            ],
            "statistics": {
                "total_messages": len(messages),
                "total_tokens": sum(msg['token_count'] or 0 for msg in messages),
                "average_tokens_per_message": sum(msg['token_count'] or 0 for msg in messages) // len(messages) if messages else 0
            }
        }
        
        json_content = json.dumps(data, indent=2)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json_content)
        
        return json_content
    
    def export_all_formats(self, conversation_id: int, base_name: Optional[str] = None):
        """Export conversation in all formats"""
        from config import EXPORTS_DIR
        
        if not base_name:
            base_name = f"conversation_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        exports_dir = Path(EXPORTS_DIR)
        exports_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Markdown
        md_path = exports_dir / f"{base_name}.md"
        self.export_to_markdown(conversation_id, md_path)
        results['markdown'] = md_path
        
        # HTML
        html_path = exports_dir / f"{base_name}.html"
        self.export_to_html(conversation_id, html_path)
        results['html'] = html_path
        
        # JSON
        json_path = exports_dir / f"{base_name}.json"
        self.export_to_json(conversation_id, json_path)
        results['json'] = json_path
        
        return results
