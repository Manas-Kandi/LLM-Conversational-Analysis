"""
Terminal User Interface for AA Microscope
Beautiful, interactive interface using Textual
"""
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Button, Static, Label, Select, Input, TextArea, DataTable, TabbedContent, TabPane
from textual.binding import Binding
from textual import events
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from datetime import datetime
from typing import Optional, List
import threading

from config import Config
from core.conversation_engine import ConversationEngine
from core.agent import AgentFactory
from storage.database import Database
from storage.models import Message, AgentRole, Conversation
from prompts.seed_library import PromptLibrary
from analysis.semantic_drift import SemanticDriftAnalyzer
from analysis.role_detection import RoleDetectionAnalyzer
from analysis.pattern_recognition import PatternRecognitionAnalyzer
from analysis.statistical import StatisticalAnalyzer


class ConversationDisplay(Static):
    """Display for live conversation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages: List[Message] = []
    
    def add_message(self, message: Message):
        """Add a message to the display"""
        self.messages.append(message)
        self.update_display()
    
    def update_display(self):
        """Update the rendered conversation"""
        if not self.messages:
            self.update("No messages yet. Start a conversation to see the dialogue unfold.")
            return
        
        output = []
        for msg in self.messages:
            agent_name = "Agent A" if msg.role == AgentRole.AGENT_A else "Agent B"
            color = "cyan" if msg.role == AgentRole.AGENT_A else "magenta"
            
            output.append(f"[bold {color}]━━━ Turn {msg.turn_number}: {agent_name} ━━━[/bold {color}]")
            output.append(f"[{color}]{msg.content}[/{color}]")
            output.append("")
        
        self.update("\n".join(output))
    
    def clear_messages(self):
        """Clear all messages"""
        self.messages = []
        self.update_display()


class StatusBar(Static):
    """Status bar showing conversation state"""
    
    status_text = reactive("Ready")
    turn_count = reactive(0)
    
    def render(self) -> str:
        return f"[bold]Status:[/bold] {self.status_text} | [bold]Turn:[/bold] {self.turn_count}"


class AAMicroscope(App):
    """Main AA Microscope Terminal Interface"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    Header {
        background: $primary;
        color: $text;
    }
    
    Footer {
        background: $primary;
    }
    
    #main_container {
        height: 100%;
    }
    
    #sidebar {
        width: 40;
        background: $panel;
        border-right: solid $primary;
    }
    
    #content {
        width: 1fr;
    }
    
    #conversation_scroll {
        height: 1fr;
        border: solid $primary;
    }
    
    #conversation_display {
        padding: 1;
    }
    
    .section_title {
        background: $primary;
        color: $text;
        padding: 1;
        text-align: center;
        text-style: bold;
    }
    
    Button {
        margin: 1 2;
    }
    
    Button.primary {
        background: $success;
    }
    
    Button.danger {
        background: $error;
    }
    
    Button.secondary {
        background: $primary;
    }
    
    #conversation_display {
        height: 1fr;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    #status_bar {
        height: 3;
        background: $panel;
        padding: 1;
        margin: 0 1;
    }
    
    Input {
        margin: 1 2;
    }
    
    Select {
        margin: 1 2;
    }
    
    .info_panel {
        background: $panel;
        padding: 1;
        margin: 1;
        height: auto;
    }
    """
    
    BINDINGS = [
        Binding("n", "new_conversation", "Start"),
        Binding("s", "stop_conversation", "Stop"),
        Binding("a", "analyze", "Analyze"),
        Binding("q", "quit", "Quit"),
    ]
    
    TITLE = "🔬 AA Microscope"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.database = Database(Config.DATABASE_PATH)
        self.current_engine: Optional[ConversationEngine] = None
        self.conversation_running = False
    
    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()
        
        with Container(id="main_container"):
            with Horizontal():
                # Sidebar with controls
                with Vertical(id="sidebar"):
                    yield Static("🎯 Quick Actions", classes="section_title")
                    
                    yield Button("▶️  Start", id="btn_new", variant="success")
                    yield Button("⏹️  Stop", id="btn_stop", variant="default")
                    yield Button("📊 Analyze", id="btn_analyze", variant="primary")
                    
                    yield Static("\n⚙️ Config", classes="section_title")
                    agent_a_short = Config.AGENT_A_MODEL.split(':')[-1].split('/')[-1][:20]
                    agent_b_short = Config.AGENT_B_MODEL.split(':')[-1].split('/')[-1][:20]
                    yield Static(f"A: {agent_a_short}", classes="info_panel")
                    yield Static(f"B: {agent_b_short}", classes="info_panel")
                    yield Static(f"Turns: {Config.DEFAULT_MAX_TURNS}", classes="info_panel")
                
                # Main content area
                with Vertical(id="content"):
                    with TabbedContent():
                        with TabPane("💬 Conversation", id="tab_conversation"):
                            yield StatusBar(id="status_bar")
                            yield ScrollableContainer(
                                ConversationDisplay(id="conversation_display"),
                                id="conversation_scroll"
                            )
                        
                        with TabPane("🎯 Prompt Selector", id="tab_prompts"):
                            yield from self._create_prompt_selector()
                        
                        with TabPane("📊 Analysis", id="tab_analysis"):
                            yield Static("Analysis results will appear here", id="analysis_results")
                        
                        with TabPane("📚 Archive", id="tab_archive"):
                            yield self._create_archive_view()
        
        yield Footer()
    
    def _create_prompt_selector(self) -> ComposeResult:
        """Create prompt selection interface"""
        categories = PromptLibrary.get_all_categories()
        
        for cat_id, cat_name in categories.items():
            prompts = PromptLibrary.get_by_category(cat_id)
            
            # Category header
            yield Static(f"\n[bold cyan]{cat_name}[/bold cyan]")
            
            # Prompts in category
            for i, prompt in enumerate(prompts):
                prompt_id = f"prompt_{cat_id}_{i}"
                btn = Button(
                    f"📝 {prompt.prompt[:60]}...",
                    id=prompt_id,
                    variant="default"
                )
                btn.metadata = {"category": cat_id, "index": i}
                yield btn
    
    def _create_archive_view(self) -> DataTable:
        """Create archive view table"""
        table = DataTable(id="archive_table")
        table.add_columns("ID", "Category", "Date", "Turns", "Status")
        return table
    
    def on_mount(self) -> None:
        """Initialize the app"""
        # Validate configuration
        if not Config.validate():
            self.exit(message="Invalid configuration. Please check your .env file.")
        
        self.sub_title = "Ready to observe emergent phenomena"
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "btn_new":
            self.action_new_conversation()
        elif button_id == "btn_stop":
            self.action_stop_conversation()
        elif button_id == "btn_analyze":
            self.action_analyze()
        elif button_id == "btn_archive":
            self.action_view_archive()
        elif button_id == "btn_stats":
            self.show_statistics()
        elif button_id and button_id.startswith("prompt_"):
            # Prompt selection
            metadata = event.button.metadata
            self.start_conversation_with_prompt(metadata["category"], metadata["index"])
    
    def action_new_conversation(self) -> None:
        """Start a new conversation"""
        if self.conversation_running:
            self.notify("Conversation already running. Stop it first.", severity="warning")
            return
        
        # Switch to prompt selector tab
        self.query_one(TabbedContent).active = "tab_prompts"
        self.notify("Select a seed prompt to begin")
    
    def start_conversation_with_prompt(self, category: str, index: int) -> None:
        """Start conversation with selected prompt"""
        if self.conversation_running:
            self.notify("Conversation already running!", severity="warning")
            return
        
        # Get prompt
        prompt_obj = PromptLibrary.get_prompt(category, index)
        
        # Update UI
        status_bar = self.query_one("#status_bar", StatusBar)
        status_bar.status_text = "Starting conversation..."
        status_bar.turn_count = 0
        
        # Clear previous conversation
        conv_display = self.query_one("#conversation_display", ConversationDisplay)
        conv_display.clear_messages()
        
        # Create conversation engine
        self.current_engine = ConversationEngine(
            seed_prompt=prompt_obj.prompt,
            category=category,
            database=self.database,
            on_message_callback=self.on_message_received
        )
        
        self.conversation_running = True
        
        # Switch to conversation tab
        self.query_one(TabbedContent).active = "tab_conversation"
        
        # Start conversation in background thread
        thread = threading.Thread(target=self.run_conversation_thread, daemon=True)
        thread.start()
        
        self.notify(f"Started: {prompt_obj.description}", severity="information")
    
    def run_conversation_thread(self) -> None:
        """Run conversation in background thread"""
        try:
            self.current_engine.run_conversation()
            
            # Update status when done
            self.call_from_thread(self.conversation_complete)
        
        except Exception as e:
            self.call_from_thread(self.conversation_error, str(e))
    
    def on_message_received(self, message: Message) -> None:
        """Callback when new message is received"""
        # Update UI from main thread
        self.call_from_thread(self.update_conversation_display, message)
    
    def update_conversation_display(self, message: Message) -> None:
        """Update conversation display with new message"""
        conv_display = self.query_one("#conversation_display", ConversationDisplay)
        conv_display.add_message(message)
        
        # Update status bar
        status_bar = self.query_one("#status_bar", StatusBar)
        status_bar.turn_count = message.turn_number
        status_bar.status_text = f"Turn {message.turn_number} - {'Agent A' if message.role == AgentRole.AGENT_A else 'Agent B'} responding..."
    
    def conversation_complete(self) -> None:
        """Handle conversation completion"""
        self.conversation_running = False
        
        status_bar = self.query_one("#status_bar", StatusBar)
        status_bar.status_text = "Conversation completed!"
        
        self.notify("Conversation finished! Run analysis to explore the results.", severity="success")
    
    def conversation_error(self, error: str) -> None:
        """Handle conversation error"""
        self.conversation_running = False
        
        status_bar = self.query_one("#status_bar", StatusBar)
        status_bar.status_text = f"Error: {error}"
        
        self.notify(f"Error: {error}", severity="error")
    
    def action_stop_conversation(self) -> None:
        """Stop the current conversation"""
        if not self.conversation_running or not self.current_engine:
            self.notify("No conversation running", severity="warning")
            return
        
        self.current_engine.stop()
        self.conversation_running = False
        
        status_bar = self.query_one("#status_bar", StatusBar)
        status_bar.status_text = "Conversation stopped"
        
        self.notify("Conversation stopped", severity="information")
    
    def action_analyze(self) -> None:
        """Run analysis on current conversation"""
        if not self.current_engine:
            self.notify("No conversation to analyze", severity="warning")
            return
        
        if self.conversation_running:
            self.notify("Wait for conversation to finish first", severity="warning")
            return
        
        self.notify("Running analysis... This may take a moment.", severity="information")
        
        # Run analysis in background
        thread = threading.Thread(target=self.run_analysis_thread, daemon=True)
        thread.start()
    
    def run_analysis_thread(self) -> None:
        """Run analysis in background thread"""
        try:
            conversation = self.current_engine.get_conversation()
            
            # Run all analyses
            results = {}
            
            # Statistical (fast, no LLM)
            stat_analyzer = StatisticalAnalyzer(conversation, self.database)
            results['statistical'] = stat_analyzer.analyze()
            
            # LLM-based analyses (slower)
            drift_analyzer = SemanticDriftAnalyzer(conversation, self.database)
            results['semantic_drift'] = drift_analyzer.analyze()
            
            role_analyzer = RoleDetectionAnalyzer(conversation, self.database)
            results['role_detection'] = role_analyzer.analyze()
            
            pattern_analyzer = PatternRecognitionAnalyzer(conversation, self.database)
            results['pattern_recognition'] = pattern_analyzer.analyze()
            
            # Update UI
            self.call_from_thread(self.display_analysis_results, results)
        
        except Exception as e:
            self.call_from_thread(self.notify, f"Analysis error: {str(e)}", severity="error")
    
    def display_analysis_results(self, results: dict) -> None:
        """Display analysis results in UI"""
        # Switch to analysis tab
        self.query_one(TabbedContent).active = "tab_analysis"
        
        # Format results
        output = []
        output.append("[bold cyan]═══ ANALYSIS RESULTS ═══[/bold cyan]\n")
        
        for analysis_type, result in results.items():
            output.append(f"[bold yellow]▸ {analysis_type.replace('_', ' ').title()}[/bold yellow]")
            output.append(f"[green]{result.summary}[/green]")
            output.append("")
        
        # Update display
        analysis_widget = self.query_one("#analysis_results", Static)
        analysis_widget.update("\n".join(output))
        
        self.notify("Analysis complete!", severity="success")
    
    def action_view_archive(self) -> None:
        """Switch to archive tab"""
        self.query_one(TabbedContent).active = "tab_archive"
    
    def action_scroll_down(self) -> None:
        """Scroll conversation down"""
        try:
            scroll_container = self.query_one("#conversation_scroll", ScrollableContainer)
            scroll_container.scroll_down()
        except:
            pass
    
    def action_scroll_up(self) -> None:
        """Scroll conversation up"""
        try:
            scroll_container = self.query_one("#conversation_scroll", ScrollableContainer)
            scroll_container.scroll_up()
        except:
            pass
    
    def show_statistics(self) -> None:
        """Show database statistics"""
        stats = self.database.get_statistics()
        
        message = f"""
📊 Database Statistics:
━━━━━━━━━━━━━━━━━━━━
Total Conversations: {stats['total_conversations']}
Total Messages: {stats['total_messages']}
Total Analyses: {stats['total_analyses']}

By Status: {stats['by_status']}
By Category: {len(stats['by_category'])} categories
        """
        
        self.notify(message, severity="information")


def launch_tui():
    """Launch the terminal user interface"""
    app = AAMicroscope()
    app.run()


if __name__ == "__main__":
    launch_tui()
