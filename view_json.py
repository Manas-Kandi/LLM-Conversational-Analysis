#!/usr/bin/env python3
"""
Simple JSON conversation viewer
Browse conversations saved as JSON files
"""
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def list_conversations():
    """List all JSON conversations"""
    json_dir = Path("conversations_json")
    
    if not json_dir.exists():
        console.print("[yellow]No conversations found. Run a test first![/yellow]")
        return
    
    files = sorted(json_dir.glob("conv_*.json"), reverse=True)
    
    if not files:
        console.print("[yellow]No conversations found. Run a test first![/yellow]")
        return
    
    table = Table(title="🔬 Saved Conversations", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Category", style="green")
    table.add_column("Turns", justify="right", width=8)
    table.add_column("Status", style="yellow")
    table.add_column("File", style="dim")
    
    for filepath in files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        table.add_row(
            str(data['id']),
            data['metadata']['category'],
            str(data['metadata']['total_turns']),
            data['metadata']['status'],
            filepath.name
        )
    
    console.print(table)
    console.print(f"\n💡 View conversation: [cyan]python view_json.py <id>[/cyan]")
    console.print(f"💡 Or open file directly: [cyan]conversations_json/conv_*.json[/cyan]")


def view_conversation(conv_id: int):
    """View a specific conversation"""
    json_dir = Path("conversations_json")
    
    # Find file with this ID
    files = list(json_dir.glob(f"conv_{conv_id}_*.json"))
    
    if not files:
        console.print(f"[red]❌ Conversation {conv_id} not found[/red]")
        return
    
    filepath = files[0]
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Display metadata
    console.print(Panel(
        f"[bold cyan]Conversation #{data['id']}[/bold cyan]\n"
        f"Category: {data['metadata']['category']}\n"
        f"Status: {data['metadata']['status']}\n"
        f"Turns: {data['metadata']['total_turns']}\n"
        f"Started: {data['metadata']['start_time']}",
        title="📋 Metadata",
        border_style="cyan"
    ))
    
    console.print()
    
    # Display seed prompt
    console.print(Panel(
        data['metadata']['seed_prompt'],
        title="🌱 Seed Prompt",
        border_style="yellow"
    ))
    
    console.print()
    
    # Display agents
    console.print(Panel(
        f"[bold blue]Agent A:[/bold blue] {data['agents']['agent_a']['model']}\n"
        f"Temperature: {data['agents']['agent_a']['temperature']}\n\n"
        f"[bold magenta]Agent B:[/bold magenta] {data['agents']['agent_b']['model']}\n"
        f"Temperature: {data['agents']['agent_b']['temperature']}",
        title="🤖 Agents",
        border_style="green"
    ))
    
    console.print()
    console.print("[bold]💬 Conversation:[/bold]")
    console.print()
    
    # Display messages
    for msg in data['messages']:
        role_color = "blue" if msg['role'] == 'agent_a' else "magenta"
        agent_name = "Agent A" if msg['role'] == 'agent_a' else "Agent B"
        
        console.print(Panel(
            f"{msg['content']}\n\n"
            f"[dim]Model: {msg['model']} | Tokens: {msg['token_count'] or 'N/A'}[/dim]",
            title=f"[{role_color}]Turn {msg['turn']}: {agent_name}[/{role_color}]",
            border_style=role_color
        ))
        console.print()
    
    # Display statistics
    stats = data['statistics']
    console.print(Panel(
        f"Total Messages: {stats['total_messages']}\n"
        f"Total Tokens: {stats['total_tokens']}\n"
        f"Avg Tokens/Message: {stats['average_tokens_per_message']}\n"
        f"Duration: {stats['duration_seconds']} seconds" if stats['duration_seconds'] else "Duration: N/A",
        title="📊 Statistics",
        border_style="green"
    ))
    
    console.print()
    console.print(f"[dim]File: {filepath}[/dim]")


def main():
    if len(sys.argv) < 2:
        list_conversations()
    else:
        try:
            conv_id = int(sys.argv[1])
            view_conversation(conv_id)
        except ValueError:
            console.print("[red]❌ Invalid conversation ID. Must be a number.[/red]")
            sys.exit(1)


if __name__ == "__main__":
    main()
