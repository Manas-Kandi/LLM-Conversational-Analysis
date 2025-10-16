#!/usr/bin/env python3
"""
Research Templates Quick Start
Interactive script to help you get started with the research system
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from research.template_executor import TemplateExecutor
from research.batch_runner import BatchRunner
import sys

console = Console()


def show_welcome():
    """Display welcome message"""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]AA Microscope Research Templates System[/bold cyan]\n\n"
        "Systematic, reproducible experiments for agent-agent dialogue research",
        border_style="cyan"
    ))
    console.print()


def list_templates_interactive():
    """Interactive template listing"""
    executor = TemplateExecutor()
    
    console.print("[bold yellow]📋 Available Research Templates[/bold yellow]\n")
    
    categories = ["All", "parameter_sweep", "model_comparison", 
                 "phenomenon_specific", "stress_test", "longitudinal"]
    
    console.print("[cyan]Select category:[/cyan]")
    for i, cat in enumerate(categories, 1):
        console.print(f"  {i}. {cat}")
    
    choice = Prompt.ask("\nChoice", choices=[str(i) for i in range(1, len(categories)+1)], default="1")
    category = None if choice == "1" else categories[int(choice)-1]
    
    templates = executor.list_templates(category)
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Description", style="white")
    table.add_column("Priority", justify="center")
    table.add_column("Runs", justify="right")
    table.add_column("Duration", justify="right")
    
    for tmpl in templates:
        priority_color = {
            "critical": "red",
            "high": "yellow", 
            "medium": "blue",
            "low": "dim"
        }.get(tmpl.get('priority', 'low'), "white")
        
        table.add_row(
            tmpl['template_id'],
            tmpl['category'],
            tmpl['description'][:50] + "..." if len(tmpl['description']) > 50 else tmpl['description'],
            f"[{priority_color}]{tmpl.get('priority', 'N/A')}[/{priority_color}]",
            str(tmpl.get('estimated_runs', '?')),
            f"{tmpl.get('estimated_duration_minutes', '?')} min"
        )
    
    console.print("\n")
    console.print(table)
    console.print()


def show_template_details(template_id: str):
    """Show detailed information about a template"""
    executor = TemplateExecutor()
    
    try:
        template = executor.get_template(template_id)
        
        console.print(f"\n[bold cyan]Template: {template_id}[/bold cyan]\n")
        console.print(f"[yellow]Category:[/yellow] {template.get('category')}")
        console.print(f"[yellow]Type:[/yellow] {template.get('type')}")
        console.print(f"[yellow]Description:[/yellow] {template.get('description')}\n")
        console.print(f"[yellow]Research Question:[/yellow] {template.get('research_question')}\n")
        
        if template.get('hypothesis'):
            console.print(f"[yellow]Hypothesis:[/yellow] {template.get('hypothesis')}\n")
        
        metadata = template.get('metadata', {})
        console.print(f"[yellow]Priority:[/yellow] {metadata.get('priority')}")
        console.print(f"[yellow]Estimated Runs:[/yellow] {metadata.get('estimated_runs')}")
        console.print(f"[yellow]Estimated Duration:[/yellow] {metadata.get('estimated_duration_minutes')} minutes\n")
        
        # Generate preview of runs
        runs = executor.generate_experiment_runs(template_id)
        console.print(f"[green]Would generate {len(runs)} experiment runs[/green]\n")
        
        if len(runs) > 0:
            console.print("[dim]First 3 runs:[/dim]")
            for i, run in enumerate(runs[:3], 1):
                console.print(f"  {i}. {run.run_id}")
                console.print(f"     Parameters: {run.parameters}")
        
        console.print()
        
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


def run_template_interactive():
    """Interactive template execution"""
    executor = TemplateExecutor()
    
    # List templates first
    templates = executor.list_templates()
    console.print("\n[bold yellow]Available Templates:[/bold yellow]")
    for i, tmpl in enumerate(templates, 1):
        priority = tmpl.get('priority', 'N/A')
        console.print(f"  {i}. {tmpl['template_id']} ([{priority}] priority)")
    
    console.print()
    template_id = Prompt.ask("Enter template ID")
    
    # Show details
    show_template_details(template_id)
    
    # Ask for execution options
    console.print("[cyan]Execution Options:[/cyan]\n")
    
    test_mode = Confirm.ask("Run in test mode (first 3 runs only)?", default=True)
    
    if not test_mode:
        max_runs = Prompt.ask("Maximum runs (leave empty for all)", default="")
        max_runs = int(max_runs) if max_runs else None
        
        parallel = Prompt.ask("Parallel execution threads", default="1")
        parallel = int(parallel)
    else:
        max_runs = 3
        parallel = 1
    
    # Confirm
    console.print()
    if not Confirm.ask(f"Execute {template_id}?", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    # Execute
    console.print(f"\n[bold green]🚀 Executing {template_id}...[/bold green]\n")
    
    if test_mode:
        # Use template executor for test
        runs = executor.generate_experiment_runs(template_id)[:max_runs]
        for run in runs:
            executor.execute_run(run)
    else:
        # Use batch runner for full execution
        runner = BatchRunner(parallel=parallel)
        result = runner.run_template_batch(template_id, max_runs=max_runs)
        
        if result:
            console.print("\n[bold green]✅ Execution complete![/bold green]")
            console.print(f"Results saved to: research_results/{result.batch_id}_results.json")


def guided_research_workflow():
    """Guided research workflow"""
    console.print("\n[bold cyan]🔬 Guided Research Workflow[/bold cyan]\n")
    
    console.print("This will help you plan a systematic research study.\n")
    
    # Phase selection
    console.print("[yellow]Research Phase:[/yellow]")
    console.print("  1. Foundational (establish baselines)")
    console.print("  2. Exploratory (discover phenomena)")
    console.print("  3. Targeted (test specific hypotheses)")
    console.print("  4. Longitudinal (extended studies)")
    
    phase = Prompt.ask("\nSelect phase", choices=["1", "2", "3", "4"], default="1")
    
    recommendations = {
        "1": ["temperature_matrix", "architecture_comparison", "identity_archaeology"],
        "2": ["emotional_contagion", "creativity_emergence", "context_window_sweep"],
        "3": ["model_size_scaling", "david_goliath", "conflict_escalation"],
        "4": ["ultra_endurance", "agent_personality_stability"]
    }
    
    console.print(f"\n[green]Recommended templates for this phase:[/green]\n")
    for template_id in recommendations[phase]:
        console.print(f"  - {template_id}")
    
    console.print("\n[yellow]Suggested workflow:[/yellow]")
    console.print("1. Run each template with --max-runs 10 for initial testing")
    console.print("2. Review results and identify interesting patterns")
    console.print("3. Run full batches on promising templates")
    console.print("4. Generate comparative reports")
    console.print()
    
    if Confirm.ask("Would you like to start execution?", default=False):
        for template_id in recommendations[phase]:
            console.print(f"\n[cyan]Execute {template_id}?[/cyan]")
            if Confirm.ask("Continue", default=True):
                runner = BatchRunner(parallel=1)
                runner.run_template_batch(template_id, max_runs=10)


def main():
    """Main menu"""
    show_welcome()
    
    while True:
        console.print("[bold cyan]Main Menu:[/bold cyan]\n")
        console.print("  1. List all templates")
        console.print("  2. Show template details")
        console.print("  3. Run a template")
        console.print("  4. Guided research workflow")
        console.print("  5. View documentation")
        console.print("  6. Exit")
        console.print()
        
        choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5", "6"], default="1")
        
        if choice == "1":
            list_templates_interactive()
        elif choice == "2":
            template_id = Prompt.ask("\nEnter template ID")
            show_template_details(template_id)
        elif choice == "3":
            run_template_interactive()
        elif choice == "4":
            guided_research_workflow()
        elif choice == "5":
            console.print("\n[yellow]Documentation:[/yellow]")
            console.print("  - Full Guide: RESEARCH_TEMPLATES_GUIDE.md")
            console.print("  - Quick Reference: research/README.md")
            console.print("  - Main README: README.md")
            console.print()
        elif choice == "6":
            console.print("\n[cyan]Happy researching! 🔬[/cyan]\n")
            break
        
        if not Confirm.ask("\nReturn to main menu?", default=True):
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Cancelled by user[/yellow]")
        sys.exit(0)
