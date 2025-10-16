"""
Export utilities for conversations and analyses
Generate research-ready reports in various formats
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import csv

from storage.models import Conversation, AnalysisResult
from storage.database import Database
from config import Config


class ConversationExporter:
    """Export conversations and analyses to various formats"""
    
    def __init__(self, database: Optional[Database] = None, output_dir: Optional[Path] = None):
        self.database = database or Database(Config.DATABASE_PATH)
        self.output_dir = output_dir or Config.EXPORTS_DIR
        self.output_dir.mkdir(exist_ok=True)
    
    def export_conversation_json(self, conversation: Conversation, filename: Optional[str] = None) -> Path:
        """Export conversation to JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{conversation.id}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(conversation.to_json())
        
        return filepath
    
    def export_conversation_markdown(self, conversation: Conversation, filename: Optional[str] = None) -> Path:
        """Export conversation to Markdown format"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{conversation.id}_{timestamp}.md"
        
        filepath = self.output_dir / filename
        
        md_lines = []
        md_lines.append(f"# Conversation {conversation.id}: {conversation.category}")
        md_lines.append("")
        md_lines.append(f"**Seed Prompt:** {conversation.seed_prompt}")
        md_lines.append("")
        md_lines.append(f"**Configuration:**")
        md_lines.append(f"- Agent A: {conversation.agent_a_model} (temp={conversation.agent_a_temp})")
        md_lines.append(f"- Agent B: {conversation.agent_b_model} (temp={conversation.agent_b_temp})")
        md_lines.append(f"- Total Turns: {conversation.total_turns}")
        md_lines.append(f"- Duration: {conversation.get_duration():.1f}s" if conversation.get_duration() else "- Duration: N/A")
        md_lines.append(f"- Status: {conversation.status}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("## Conversation")
        md_lines.append("")
        
        for msg in conversation.messages:
            agent_name = "Agent A" if msg.role.value == "agent_a" else "Agent B"
            md_lines.append(f"### Turn {msg.turn_number}: {agent_name}")
            md_lines.append("")
            md_lines.append(msg.content)
            md_lines.append("")
        
        with open(filepath, 'w') as f:
            f.write("\n".join(md_lines))
        
        return filepath
    
    def export_conversation_csv(self, conversation: Conversation, filename: Optional[str] = None) -> Path:
        """Export conversation messages to CSV"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{conversation.id}_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Turn', 'Agent', 'Content', 'Timestamp', 'Model', 'Temperature', 'Tokens'])
            
            for msg in conversation.messages:
                agent_name = "Agent A" if msg.role.value == "agent_a" else "Agent B"
                writer.writerow([
                    msg.turn_number,
                    agent_name,
                    msg.content,
                    msg.timestamp.isoformat(),
                    msg.model,
                    msg.temperature,
                    msg.token_count or ''
                ])
        
        return filepath
    
    def export_analysis_report(self, conversation_id: int, filename: Optional[str] = None) -> Path:
        """Export complete analysis report with all analyses"""
        # Get conversation
        conversation = self.database.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Get all analyses
        analyses = self.database.get_analyses(conversation_id)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_report_{conversation_id}_{timestamp}.md"
        
        filepath = self.output_dir / filename
        
        md_lines = []
        md_lines.append(f"# Analysis Report: Conversation {conversation_id}")
        md_lines.append("")
        md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("")
        md_lines.append("## Conversation Overview")
        md_lines.append("")
        md_lines.append(f"**Seed Prompt:** {conversation.seed_prompt}")
        md_lines.append("")
        md_lines.append(f"**Category:** {conversation.category}")
        md_lines.append(f"**Agent A:** {conversation.agent_a_model} (temp={conversation.agent_a_temp})")
        md_lines.append(f"**Agent B:** {conversation.agent_b_model} (temp={conversation.agent_b_temp})")
        md_lines.append(f"**Total Turns:** {conversation.total_turns}")
        md_lines.append(f"**Duration:** {conversation.get_duration():.1f}s" if conversation.get_duration() else "**Duration:** N/A")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # Add each analysis
        for analysis in analyses:
            md_lines.append(f"## {analysis.analysis_type.replace('_', ' ').title()}")
            md_lines.append("")
            md_lines.append(f"**Summary:** {analysis.summary}")
            md_lines.append("")
            md_lines.append("### Detailed Results")
            md_lines.append("")
            md_lines.append("```json")
            md_lines.append(json.dumps(analysis.results, indent=2))
            md_lines.append("```")
            md_lines.append("")
        
        # Add conversation excerpt
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("## Conversation Excerpt")
        md_lines.append("")
        md_lines.append("(First 5 and last 5 turns)")
        md_lines.append("")
        
        # First 5 turns
        for msg in conversation.messages[:5]:
            agent_name = "Agent A" if msg.role.value == "agent_a" else "Agent B"
            md_lines.append(f"**Turn {msg.turn_number} - {agent_name}:**")
            md_lines.append(f"> {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
            md_lines.append("")
        
        if len(conversation.messages) > 10:
            md_lines.append("*(... middle turns omitted ...)*")
            md_lines.append("")
            
            # Last 5 turns
            for msg in conversation.messages[-5:]:
                agent_name = "Agent A" if msg.role.value == "agent_a" else "Agent B"
                md_lines.append(f"**Turn {msg.turn_number} - {agent_name}:**")
                md_lines.append(f"> {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
                md_lines.append("")
        
        with open(filepath, 'w') as f:
            f.write("\n".join(md_lines))
        
        return filepath
    
    def export_comparative_report(self, conversation_ids: list, filename: Optional[str] = None) -> Path:
        """Export comparative analysis of multiple conversations"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparative_report_{timestamp}.md"
        
        filepath = self.output_dir / filename
        
        conversations = [self.database.get_conversation(cid) for cid in conversation_ids]
        conversations = [c for c in conversations if c]  # Filter out None
        
        md_lines = []
        md_lines.append(f"# Comparative Analysis Report")
        md_lines.append("")
        md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append(f"**Conversations Compared:** {len(conversations)}")
        md_lines.append("")
        
        # Summary table
        md_lines.append("## Conversation Summary")
        md_lines.append("")
        md_lines.append("| ID | Category | Agent A | Agent B | Turns | Duration |")
        md_lines.append("|----|----------|---------|---------|-------|----------|")
        
        for conv in conversations:
            duration = f"{conv.get_duration():.1f}s" if conv.get_duration() else "N/A"
            md_lines.append(f"| {conv.id} | {conv.category} | {conv.agent_a_model} | {conv.agent_b_model} | {conv.total_turns} | {duration} |")
        
        md_lines.append("")
        
        # Individual conversation details
        for conv in conversations:
            md_lines.append(f"## Conversation {conv.id}: {conv.category}")
            md_lines.append("")
            md_lines.append(f"**Seed:** {conv.seed_prompt[:100]}...")
            md_lines.append("")
            
            # Get analyses for this conversation
            analyses = self.database.get_analyses(conv.id)
            if analyses:
                md_lines.append("**Analyses:**")
                for analysis in analyses:
                    md_lines.append(f"- *{analysis.analysis_type}*: {analysis.summary}")
                md_lines.append("")
        
        with open(filepath, 'w') as f:
            f.write("\n".join(md_lines))
        
        return filepath
    
    def export_research_dataset(self, category: Optional[str] = None, filename: Optional[str] = None) -> Path:
        """Export all conversations as research dataset (JSON)"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cat_str = f"_{category}" if category else "_all"
            filename = f"research_dataset{cat_str}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Get conversations
        conversations = self.database.list_conversations(category=category, limit=1000)
        
        # Build dataset
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "category_filter": category,
                "total_conversations": len(conversations)
            },
            "conversations": []
        }
        
        for conv_summary in conversations:
            conv = self.database.get_conversation(conv_summary["id"])
            if conv:
                analyses = self.database.get_analyses(conv.id)
                
                dataset["conversations"].append({
                    "id": conv.id,
                    "seed_prompt": conv.seed_prompt,
                    "category": conv.category,
                    "metadata": {
                        "agent_a_model": conv.agent_a_model,
                        "agent_b_model": conv.agent_b_model,
                        "agent_a_temp": conv.agent_a_temp,
                        "agent_b_temp": conv.agent_b_temp,
                        "total_turns": conv.total_turns,
                        "duration_seconds": conv.get_duration(),
                        "status": conv.status
                    },
                    "messages": [msg.to_dict() for msg in conv.messages],
                    "analyses": [a.to_dict() for a in analyses]
                })
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return filepath
