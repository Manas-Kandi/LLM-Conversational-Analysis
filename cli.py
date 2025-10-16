#!/usr/bin/env python3
"""
Command-line interface for AA Microscope
For scripted/automated usage without TUI
"""
import click
from pathlib import Path
import sys

from config import Config
from core.conversation_engine import ConversationEngine
from core.agent import AgentFactory
from storage.database import Database
from prompts.seed_library import PromptLibrary
from analysis.semantic_drift import SemanticDriftAnalyzer
from analysis.role_detection import RoleDetectionAnalyzer
from analysis.pattern_recognition import PatternRecognitionAnalyzer
from analysis.statistical import StatisticalAnalyzer
from exports.exporter import ConversationExporter


@click.group()
def cli():
    """🔬 AA Microscope - Agent-Agent Conversation Observatory"""
    if not Config.validate():
        click.echo("❌ Invalid configuration. Check your .env file.", err=True)
        sys.exit(1)


@cli.command()
@click.option('--category', '-c', help='Prompt category')
@click.option('--index', '-i', type=int, help='Prompt index within category')
@click.option('--custom', help='Custom seed prompt text')
@click.option('--max-turns', '-t', type=int, default=None, help='Max turns (overrides config)')
@click.option('--agent-a-model', help='Override Agent A model')
@click.option('--agent-b-model', help='Override Agent B model')
@click.option('--output', '-o', help='Output file path')
def run(category, index, custom, max_turns, agent_a_model, agent_b_model, output):
    """Run a conversation"""
    
    # Determine seed prompt
    if custom:
        seed_prompt = custom
        cat = "custom"
    elif category and index is not None:
        try:
            prompt_obj = PromptLibrary.get_prompt(category, index)
            seed_prompt = prompt_obj.prompt
            cat = category
            click.echo(f"📝 Selected: {prompt_obj.description}")
        except (IndexError, KeyError):
            click.echo(f"❌ Invalid prompt: {category}[{index}]", err=True)
            sys.exit(1)
    else:
        click.echo("❌ Provide either --category and --index, or --custom", err=True)
        sys.exit(1)
    
    click.echo(f"\n🔬 Starting conversation...")
    click.echo(f"Seed: {seed_prompt[:80]}...")
    
    # Create agents
    agent_a = AgentFactory.create_agent_a(model=agent_a_model) if agent_a_model else None
    agent_b = AgentFactory.create_agent_b(model=agent_b_model) if agent_b_model else None
    
    # Create engine
    engine = ConversationEngine(
        seed_prompt=seed_prompt,
        category=cat,
        agent_a=agent_a,
        agent_b=agent_b,
        max_turns=max_turns
    )
    
    # Run conversation
    def on_message(msg):
        agent = "A" if msg.role.value == "agent_a" else "B"
        click.echo(f"\n[Turn {msg.turn_number} - Agent {agent}]")
        click.echo(msg.content[:200] + ("..." if len(msg.content) > 200 else ""))
    
    engine.on_message_callback = on_message
    
    conversation = engine.run_conversation()
    
    click.echo(f"\n✅ Conversation complete!")
    click.echo(f"Total turns: {conversation.total_turns}")
    click.echo(f"Status: {conversation.status}")
    
    # Export if requested
    if output:
        exporter = ConversationExporter()
        filepath = exporter.export_conversation_markdown(conversation, output)
        click.echo(f"📄 Exported to: {filepath}")
    
    click.echo(f"\n💾 Conversation saved with ID: {conversation.id}")


@cli.command()
@click.argument('conversation_id', type=int)
@click.option('--types', '-t', multiple=True, help='Analysis types (statistical, semantic_drift, role_detection, pattern_recognition)')
@click.option('--output', '-o', help='Output report file path')
def analyze(conversation_id, types, output):
    """Analyze a conversation"""
    
    # Get conversation
    db = Database(Config.DATABASE_PATH)
    conversation = db.get_conversation(conversation_id)
    
    if not conversation:
        click.echo(f"❌ Conversation {conversation_id} not found", err=True)
        sys.exit(1)
    
    click.echo(f"🔬 Analyzing conversation {conversation_id}...")
    
    # Determine which analyses to run
    analysis_types = list(types) if types else ['statistical', 'semantic_drift', 'role_detection', 'pattern_recognition']
    
    results = {}
    
    for analysis_type in analysis_types:
        click.echo(f"  Running {analysis_type}...")
        
        if analysis_type == 'statistical':
            analyzer = StatisticalAnalyzer(conversation, db)
        elif analysis_type == 'semantic_drift':
            analyzer = SemanticDriftAnalyzer(conversation, db)
        elif analysis_type == 'role_detection':
            analyzer = RoleDetectionAnalyzer(conversation, db)
        elif analysis_type == 'pattern_recognition':
            analyzer = PatternRecognitionAnalyzer(conversation, db)
        else:
            click.echo(f"  ⚠️  Unknown analysis type: {analysis_type}")
            continue
        
        result = analyzer.analyze()
        results[analysis_type] = result
        click.echo(f"  ✅ {result.summary}")
    
    click.echo(f"\n✅ Analysis complete!")
    
    # Export report if requested
    if output:
        exporter = ConversationExporter()
        filepath = exporter.export_analysis_report(conversation_id, output)
        click.echo(f"📄 Report exported to: {filepath}")


@cli.command()
@click.option('--category', '-c', help='Filter by category')
@click.option('--status', '-s', help='Filter by status')
@click.option('--limit', '-l', type=int, default=20, help='Max results')
def list(category, status, limit):
    """List conversations"""
    
    db = Database(Config.DATABASE_PATH)
    conversations = db.list_conversations(category=category, status=status, limit=limit)
    
    click.echo(f"📚 Found {len(conversations)} conversations:\n")
    click.echo(f"{'ID':<6} {'Category':<20} {'Date':<12} {'Turns':<7} {'Status':<12}")
    click.echo("=" * 70)
    
    for conv in conversations:
        click.echo(
            f"{conv['id']:<6} "
            f"{conv['category']:<20} "
            f"{conv['start_time'][:10]:<12} "
            f"{conv['total_turns']:<7} "
            f"{conv['status']:<12}"
        )


@cli.command()
@click.argument('conversation_id', type=int)
@click.option('--format', '-f', type=click.Choice(['json', 'markdown', 'csv']), default='markdown')
@click.option('--output', '-o', help='Output file path')
def export(conversation_id, format, output):
    """Export a conversation"""
    
    db = Database(Config.DATABASE_PATH)
    conversation = db.get_conversation(conversation_id)
    
    if not conversation:
        click.echo(f"❌ Conversation {conversation_id} not found", err=True)
        sys.exit(1)
    
    exporter = ConversationExporter()
    
    if format == 'json':
        filepath = exporter.export_conversation_json(conversation, output)
    elif format == 'markdown':
        filepath = exporter.export_conversation_markdown(conversation, output)
    elif format == 'csv':
        filepath = exporter.export_conversation_csv(conversation, output)
    
    click.echo(f"✅ Exported to: {filepath}")


@cli.command()
def prompts():
    """List available seed prompts"""
    
    categories = PromptLibrary.get_all_categories()
    
    click.echo("📝 Available Seed Prompts:\n")
    
    for cat_id, cat_name in categories.items():
        click.echo(f"\n[bold cyan]{cat_name}[/bold cyan] (category: {cat_id})")
        prompts = PromptLibrary.get_by_category(cat_id)
        
        for i, prompt in enumerate(prompts):
            click.echo(f"  [{i}] {prompt.prompt[:70]}...")
            click.echo(f"      → {prompt.description}")


@cli.command()
def stats():
    """Show database statistics"""
    
    db = Database(Config.DATABASE_PATH)
    stats = db.get_statistics()
    
    click.echo("📊 Database Statistics:\n")
    click.echo(f"Total Conversations: {stats['total_conversations']}")
    click.echo(f"Total Messages: {stats['total_messages']}")
    click.echo(f"Total Analyses: {stats['total_analyses']}")
    
    click.echo(f"\nBy Status:")
    for status, count in stats['by_status'].items():
        click.echo(f"  {status}: {count}")
    
    click.echo(f"\nBy Category:")
    for category, count in stats['by_category'].items():
        click.echo(f"  {category}: {count}")


@cli.command()
@click.option('--category', '-c', help='Export only this category')
@click.option('--output', '-o', help='Output file path')
def dataset(category, output):
    """Export research dataset"""
    
    exporter = ConversationExporter()
    filepath = exporter.export_research_dataset(category=category, filename=output)
    
    click.echo(f"✅ Dataset exported to: {filepath}")


if __name__ == '__main__':
    cli()
