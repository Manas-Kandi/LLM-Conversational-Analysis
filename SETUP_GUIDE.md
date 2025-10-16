# 🚀 Setup Guide - AA Microscope

Complete guide to getting started with the Agent-Agent Conversation Observatory.

---

## 📋 Prerequisites

- **Python 3.8+** installed
- **API Keys** for at least one LLM provider:
  - OpenAI API key (for GPT models)
  - Anthropic API key (for Claude models)
- **Terminal** with good text rendering (for TUI)

---

## 🔧 Installation

### Step 1: Navigate to Project Directory

```bash
cd /Users/manaskandimalla/Desktop/Projects/pekoflabs
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `textual` - Beautiful terminal UI
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `sqlalchemy` - Database management
- `rich` - Rich text formatting
- And more...

### Step 4: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your preferred text editor:

```bash
nano .env
# or
vim .env
# or
code .env  # If you have VS Code
```

**Required configuration:**

```env
# At minimum, add one API key
OPENAI_API_KEY=sk-your-actual-openai-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here

# Optionally customize agent models
AGENT_A_MODEL=gpt-4
AGENT_B_MODEL=gpt-4

# Or use Claude
# AGENT_A_MODEL=claude-3-opus-20240229
# AGENT_B_MODEL=claude-3-opus-20240229

# Adjust conversation parameters
DEFAULT_MAX_TURNS=30
CONTEXT_WINDOW_SIZE=10
```

### Step 5: Test Configuration

```bash
python main.py
```

If configuration is valid, the terminal UI will launch! 🎉

---

## 🎮 Usage Modes

### Mode 1: Terminal UI (Recommended for Interactive Use)

**Launch:**
```bash
python main.py
```

**Features:**
- 📱 Beautiful, interactive interface
- 🔴 Real-time conversation viewing
- 📊 Built-in analysis dashboard
- 📚 Prompt library browser
- 🗃️ Archive explorer

**Controls:**
- `n` - New conversation
- `s` - Stop conversation
- `a` - Analyze current conversation
- `v` - View archive
- `q` - Quit

**Workflow:**
1. Press `n` or click "New Conversation"
2. Select a seed prompt from the library
3. Watch the conversation unfold in real-time
4. Press `a` to analyze when complete
5. View results in the Analysis tab

---

### Mode 2: Command-Line Interface (For Automation)

**List available prompts:**
```bash
python cli.py prompts
```

**Run a conversation:**
```bash
# Using library prompt
python cli.py run --category identity --index 0 --max-turns 20

# Using custom prompt
python cli.py run --custom "Your custom seed prompt here" --max-turns 15

# With output
python cli.py run --category emotional --index 0 --output my_conversation.md
```

**Analyze a conversation:**
```bash
python cli.py analyze 1

# Specific analyses only
python cli.py analyze 1 --types statistical --types semantic_drift

# With output report
python cli.py analyze 1 --output analysis_report.md
```

**List conversations:**
```bash
python cli.py list

# Filter by category
python cli.py list --category identity

# Filter by status
python cli.py list --status completed --limit 50
```

**Export conversations:**
```bash
# Export as markdown
python cli.py export 1 --format markdown

# Export as JSON
python cli.py export 1 --format json --output conv_1.json

# Export as CSV
python cli.py export 1 --format csv
```

**Database statistics:**
```bash
python cli.py stats
```

**Export research dataset:**
```bash
# All conversations
python cli.py dataset --output full_dataset.json

# Specific category
python cli.py dataset --category identity --output identity_dataset.json
```

---

### Mode 3: Python API (For Custom Scripts)

**Quick start example:**
```bash
python examples/quick_start.py
```

**Custom script:**
```python
from core.conversation_engine import ConversationEngine

# Create engine
engine = ConversationEngine(
    seed_prompt="Your prompt here",
    category="custom",
    max_turns=20
)

# Run conversation
conversation = engine.run_conversation()

print(f"Completed {conversation.total_turns} turns")
print(f"Conversation ID: {conversation.id}")
```

**Run batch experiments:**
```bash
python examples/batch_experiment.py
```

**Temperature sweep:**
```bash
python examples/temperature_sweep.py
```

---

## 📊 Analysis Types

### 1. Statistical Analysis (Fast, No LLM)
- Message counts and lengths
- Token usage
- Vocabulary diversity
- Agent comparison metrics
- Timing analysis

### 2. Semantic Drift Analysis (LLM-based)
- Topic evolution from seed prompt
- Turn-by-turn relevance scores
- Drift rate and trajectory
- Major topic shifts

### 3. Role Detection (LLM-based)
- Emergent persona identification
- Power dynamics
- Role stability
- AI self-awareness detection

### 4. Pattern Recognition (LLM-based)
- Recurring conversational patterns
- Emergent phenomena
- Creativity assessment
- Information dynamics
- Notable moments

---

## 📁 Project Structure

```
pekoflabs/
├── main.py                  # TUI entry point
├── cli.py                   # CLI entry point
├── config.py                # Configuration
├── requirements.txt         # Dependencies
├── .env                     # Your API keys (don't commit!)
├── README.md               # Project overview
├── SETUP_GUIDE.md          # This file
│
├── core/                    # Core conversation engine
│   ├── agent.py            # Agent wrapper
│   ├── conversation_engine.py  # Main engine
│   └── context_manager.py  # Context handling
│
├── storage/                 # Data persistence
│   ├── database.py         # SQLite operations
│   ├── models.py           # Data models
│   └── conversations.db    # Database (auto-created)
│
├── analysis/               # Analysis modules
│   ├── analyzer.py         # Base analyzer
│   ├── semantic_drift.py   # Drift analysis
│   ├── role_detection.py   # Role analysis
│   ├── pattern_recognition.py  # Pattern analysis
│   └── statistical.py      # Statistical analysis
│
├── interface/              # User interfaces
│   └── tui.py             # Textual TUI
│
├── prompts/                # Seed prompt library
│   └── seed_library.py    # 40+ research prompts
│
├── exports/                # Export utilities
│   └── exporter.py        # Various export formats
│
├── examples/               # Example scripts
│   ├── quick_start.py     # Simple example
│   ├── batch_experiment.py  # Batch runner
│   └── temperature_sweep.py  # Temperature testing
│
└── logs/                   # Log files (auto-created)
```

---

## 🔬 Research Workflow

### Typical Research Session

1. **Explore Prompts**
   ```bash
   python cli.py prompts
   ```

2. **Run Conversations**
   ```bash
   # Start with interesting prompt
   python main.py  # Interactive
   # or
   python cli.py run --category identity --index 0
   ```

3. **Analyze Results**
   ```bash
   python cli.py analyze <conversation_id>
   ```

4. **Export for Research**
   ```bash
   python cli.py export <conversation_id> --format markdown
   ```

5. **Compare Multiple Conversations**
   - Export dataset
   - Generate comparative reports
   - Identify patterns across conversations

### Experimental Designs

**Cross-Model Comparison:**
```bash
# Run same prompt with different models
python cli.py run --category meta_cognition --index 0 \
    --agent-a-model gpt-4 --agent-b-model claude-3-opus-20240229
```

**Temperature Sweep:**
```bash
python examples/temperature_sweep.py
```

**Category-Wide Study:**
```bash
python examples/batch_experiment.py
```

---

## 💡 Tips & Best Practices

### Cost Management
- Start with **shorter conversations** (5-10 turns) for testing
- Use **GPT-3.5-turbo** for cheaper experiments
- Monitor token usage in analysis results
- Set reasonable `DEFAULT_MAX_TURNS` in `.env`

### Interesting Research Questions
1. **Which categories show highest semantic drift?**
2. **Do agents ever "realize" they're talking to another AI?**
3. **How does temperature affect role stability?**
4. **Do certain prompts lead to conversational collapse?**
5. **What patterns emerge in long conversations (50+ turns)?**

### Data Organization
- Use **meaningful category names**
- Add **metadata** to conversations for later analysis
- **Export regularly** to avoid data loss
- Keep **research journal** of interesting findings

### Analysis Strategy
- Run **statistical analysis first** (it's fast)
- Use **LLM-based analyses** for deeper insights
- **Compare across conversations** for meta-patterns
- Look for **unexpected phenomena** (that's where the gold is!)

---

## 🐛 Troubleshooting

### "Configuration Error"
- Check that `.env` file exists
- Verify API keys are correct (no quotes needed)
- Ensure at least one API key is set

### "Module not found"
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### "API Error: Rate limit"
- Add delays between conversations
- Lower `max_turns` value
- Use different model tier

### "Database locked"
- Only one TUI instance can run at a time
- Close other instances
- Delete `.db-journal` file if stuck

### Terminal UI looks broken
- Ensure terminal supports Unicode/emojis
- Try different terminal emulator
- Update Textual: `pip install --upgrade textual`

---

## 🔐 Security Notes

- ⚠️ **Never commit `.env` file** to version control
- 🔒 Keep API keys private
- 💰 Monitor API usage/costs
- 🗑️ Delete sensitive conversations if needed

---

## 📚 Further Reading

**Research Inspiration:**
- Emergent phenomena in multi-agent systems
- Conversational AI alignment research
- Theory of Mind in language models
- Semantic drift in dialogue systems

**Technical References:**
- OpenAI API documentation
- Anthropic Claude documentation
- Textual TUI framework docs
- SQLite best practices

---

## 🤝 Contributing Ideas

Potential enhancements:
- Additional analysis types (sentiment, toxicity, etc.)
- Visualization dashboards (Plotly, Matplotlib)
- Real-time intervention mode
- Multi-agent (3+ agents) conversations
- Cross-language experiments
- Voice synthesis for conversations
- Integration with other LLM providers

---

## 🎓 Academic Use

If you use this for research:
- Document your methodology
- Report model versions and parameters
- Share interesting findings!
- Consider publishing datasets (with appropriate anonymization)

---

## 🎉 You're Ready!

```bash
# Start exploring!
python main.py
```

**Happy researching! 🔬**
