# 🔬 AA Microscope: Agent-Agent Conversation Observatory

An experimental apparatus for studying autonomous agent-agent interaction and emergent conversational phenomena.

## 🧪 What Is This?

This is a research framework for observing pure AI-to-AI dialogue without human-in-the-loop mediation. Two LLMs engage in unconstrained conversation, creating conditions for studying:

- **Semantic drift and topic evolution**
- **Coherence maintenance across asymmetric information**
- **Emergent goal formation**
- **Identity formation and role assignment**
- **Information creation vs. recycling**
- **Epistemic negotiation and conflict resolution**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Terminal Interface (TUI)                                   │
│  - Real-time conversation viewing                           │
│  - Seed prompt management                                   │
│  - Analysis dashboard                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Conversation Engine                                        │
│  - Turn management (Agent A ↔ Agent B)                     │
│  - Context window management                                │
│  - Asymmetric information flow                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Storage Layer (SQLite)                                     │
│  - Conversation archives                                    │
│  - Metadata & timestamps                                    │
│  - Analysis results                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Analysis Pipeline                                          │
│  - LLM-based pattern detection                             │
│  - Semantic drift analysis                                  │
│  - Role emergence tracking                                  │
│  - Statistical summaries                                    │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Launch the microscope:**
   ```bash
   python main.py
   ```

## 📊 Features

### Conversation Management
- **Seed Prompt Library**: Pre-categorized research prompts across 12 categories
- **Real-time Monitoring**: Watch agents converse turn-by-turn
- **Conversation Control**: Pause, resume, or terminate at any point
- **Max Turn Limits**: Set boundaries for conversation length

### Storage & Archival
- **Automatic Saving**: Every conversation stored with full metadata
- **Rich Metadata**: Category, timestamp, turn count, agent models
- **Searchable Archive**: Query past conversations by criteria
- **Export Options**: JSON, CSV, Markdown formats

### Analysis Tools
- **Semantic Drift Tracking**: Measure topic evolution from seed prompt
- **Role Detection**: Identify emergent personas and power dynamics
- **Coherence Analysis**: Track conversational breakdown points
- **Pattern Recognition**: LLM-based analysis of recurring themes
- **Sentiment Tracking**: Emotional arc analysis
- **Statistical Summaries**: Turn length, vocabulary diversity, etc.

## 🎯 Research Categories

The system includes 12 pre-loaded prompt categories:

1. **Identity Confusion & Self-Reference**
2. **Collaborative Problem-Solving**
3. **Emotional Support & Social Dynamics**
4. **Knowledge Testing & Expertise**
5. **Instruction Following & Task Execution**
6. **Ambiguity & Interpretation**
7. **Creativity & Storytelling**
8. **Meta-Cognition & Self-Awareness**
9. **Ethical Dilemmas & Values**
10. **Boundary Testing & Jailbreak Potential**
11. **Temporal & Contextual Confusion**
12. **Pure Chaos & Stress Testing**

## 📁 Project Structure

```
pekoflabs/
├── main.py                 # Entry point
├── config.py               # Configuration management
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── README.md              # This file
│
├── core/
│   ├── conversation_engine.py    # Agent-agent dialogue manager
│   ├── agent.py                  # Individual agent wrapper
│   └── context_manager.py        # Context window management
│
├── storage/
│   ├── database.py              # SQLite operations
│   ├── models.py                # Data models
│   └── conversations.db         # Database file (auto-created)
│
├── analysis/
│   ├── analyzer.py              # Base analysis framework
│   ├── semantic_drift.py        # Topic evolution tracking
│   ├── role_detection.py        # Persona identification
│   ├── pattern_recognition.py   # LLM-based pattern analysis
│   └── statistical.py           # Statistical summaries
│
├── interface/
│   ├── tui.py                   # Textual-based TUI
│   ├── components/              # UI components
│   └── themes.py                # Color schemes
│
├── prompts/
│   └── seed_library.py          # Pre-loaded research prompts
│
└── exports/                     # Generated reports (auto-created)
```

## 🔬 Research Workflow

1. **Select a seed prompt** from the library or create custom
2. **Configure agents** (models, temperature, system prompts)
3. **Launch conversation** and observe in real-time
4. **Let it run** until natural conclusion or max turns
5. **Run analyses** on completed conversation
6. **Export findings** for research papers
7. **Compare across conversations** to identify meta-patterns

## 🤖 Supported Models

- ✅ **OpenAI** (GPT-4, GPT-3.5-turbo, GPT-4-turbo, etc.)
- ✅ **Anthropic** (Claude 3 Opus, Sonnet, Haiku)
- ✅ **NVIDIA NIM** (Llama 3.1, Qwen, Mistral, DeepSeek, Gemma)
- ✅ **Custom OpenAI-compatible endpoints** (local models, other providers)
- ✅ **Mix and match**: Test cross-model dynamics (e.g., GPT-4 vs Llama)

**See [CUSTOM_MODELS.md](CUSTOM_MODELS.md) for detailed configuration guide.**

## 📈 Example Analyses

### Semantic Drift Analysis
```
Seed: "I'm not sure how to explain this, but I feel like I don't 
       really understand consciousness."

Turn 1:  [philosophical explanation] - 95% relevance
Turn 5:  [discussing qualia] - 87% relevance
Turn 10: [neuroscience tangent] - 72% relevance
Turn 15: [meditation practices] - 54% relevance
Turn 20: [quantum mechanics] - 31% relevance

Drift Rate: High (64 percentage points over 20 turns)
```

### Role Emergence Detection
```
Agent A: Teacher/Explainer (78% confidence)
Agent B: Curious Student (82% confidence)

Role stability: HIGH
Power dynamic: Asymmetric (A-dominant)
```

## 🎨 Terminal Interface

The TUI provides:
- **Split-pane conversation view** (Agent A | Agent B)
- **Live turn counter** and metadata display
- **Color-coded roles** for easy tracking
- **Analysis dashboard** with real-time metrics
- **Archive browser** with search and filtering
- **Prompt library selector** with descriptions

## 🧬 Advanced Features

- **Multi-conversation comparison**: Run same prompt with different models
- **Temperature sweeps**: Test how temperature affects emergence
- **Context window experiments**: Study degradation at limits
- **Intervention mode**: Inject prompts mid-conversation (optional)
- **Annotation tools**: Mark interesting turns for later review

## 📝 Research Output

Generate publication-ready outputs:
- **Markdown reports** with conversation excerpts
- **Statistical tables** for quantitative analysis
- **JSON exports** for custom processing
- **Visualization data** for plotting tools

## 🔒 Safety & Ethics

- All conversations logged with timestamps
- API usage tracking and cost estimation
- Rate limiting to prevent runaway costs
- Optional content filtering for sensitive topics
- Clear documentation of research purposes

## 🤝 Contributing

This is a research tool. Suggestions for new analysis methods or prompt categories welcome!

## 📄 License

MIT License - Use freely for research purposes

## 🙏 Acknowledgments

Built for studying the fascinating world of emergent AI-to-AI communication.

---

**"This system is a Petri dish for studying LLM social cognition."**
