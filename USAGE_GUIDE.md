# 🚀 AA Microscope - Complete Usage Guide

Quick reference for all features.

---

## 🎯 Three Ways to Use AA Microscope

### 1. 📱 Simplified TUI (Best for Live Viewing)

```bash
python main.py
```

**Controls:**
- Press `n` to start a conversation
- Press `s` to stop
- Press `a` to analyze
- Press `q` to quit
- **Scroll**: Use mouse, arrow keys, or Page Up/Down

**What's New:**
- ✅ Cleaner interface (3 buttons instead of 5)
- ✅ Shorter model names
- ✅ Simplified footer
- ✅ Better scrolling

---

### 2. 🧪 Test Templates (Best for Research)

```bash
# List all templates
python test_templates.py list

# Run a template
python test_templates.py quick
python test_templates.py identity_probe
python test_templates.py emotional_cascade
```

**Available Templates:**
- `quick` - 5 turns (debugging)
- `standard` - 15 turns (normal)
- `extended` - 30 turns (deep analysis)
- `identity_probe` - 20 turns (self-awareness)
- `emotional_cascade` - 20 turns (empathy)
- `problem_solving` - 25 turns (collaboration)
- `chaos_test` - 15 turns (boundary testing)
- `cross_model` - 20 turns (model comparison)

**Each template automatically:**
- ✅ Runs the conversation
- ✅ Performs quantitative analysis
- ✅ Saves JSON report to `exports/`
- ✅ Stores in database

---

### 3. 🌐 Web Viewer (Best for Browsing)

```bash
python viewer.py
```

Then open: **http://localhost:5000**

**Features:**
- ✅ Beautiful table view of all conversations
- ✅ Search by category, model, or prompt
- ✅ Click any row to see full conversation
- ✅ Live statistics (conversations, messages, tokens)
- ✅ Auto-refresh every 10 seconds
- ✅ Color-coded agents (blue/purple)

---

## 📊 Quantitative Analysis Metrics

Every test template generates comprehensive metrics:

### Conversation Dynamics
- Turn-taking balance
- Response lengths
- Information entropy
- Lexical diversity

### Linguistic Analysis
- Sentence complexity
- Word complexity
- Readability scores
- Grade level

### Identity & Role
- First-person pronoun usage
- Self-reference patterns
- Uncertainty markers
- Confidence indicators

### Social Dynamics
- Directive statements
- Empathy markers
- Agreement patterns
- Question frequency

### Content Analysis
- Citation frequency
- Creative elements
- Meta-cognitive markers
- Knowledge patterns

---

## 🔬 Research Workflows

### Quick Test (5 minutes)

```bash
# 1. Run a quick test
python test_templates.py quick

# 2. View results
python viewer.py
# Open http://localhost:5000

# 3. Check JSON report
cat exports/test_quick_*.json
```

### Full Experiment (30 minutes)

```bash
# 1. Run multiple templates
python test_templates.py identity_probe
python test_templates.py emotional_cascade
python test_templates.py problem_solving

# 2. Browse in web viewer
python viewer.py

# 3. Compare JSON reports
ls -la exports/
```

### Cross-Model Study

```bash
# Edit .env for different model pairs
nano .env

# Test 1: Llama vs Llama
AGENT_A_MODEL=nvidia:meta/llama-3.1-70b-instruct
AGENT_B_MODEL=nvidia:meta/llama-3.1-8b-instruct
python test_templates.py cross_model

# Test 2: Llama vs Mistral
AGENT_A_MODEL=nvidia:meta/llama-3.1-70b-instruct
AGENT_B_MODEL=nvidia:mistralai/mixtral-8x7b-instruct-v0.1
python test_templates.py cross_model

# Compare results in viewer
python viewer.py
```

---

## 📁 File Structure

```
pekoflabs/
├── main.py                    # Simplified TUI
├── viewer.py                  # Web viewer
├── test_templates.py          # Pre-configured tests
├── storage/
│   ├── conversations.db       # SQLite database
│   └── export.py             # Export utilities
├── exports/
│   ├── test_*.json           # Quantitative reports
│   ├── conversation_*.html   # Beautiful HTML exports
│   └── conversation_*.md     # Markdown exports
├── analysis/
│   └── quantitative.py       # Metrics framework
└── templates/
    └── viewer.html           # Web UI
```

---

## 🎨 What Changed

### TUI Simplification

**Before:**
```
🔬 Controls
🆕 New Conversation
⏸️  Stop Conversation  
📊 Run Analysis
📚 View Archive
📈 Statistics

⚙️ Configuration
Agent A: nvidia:meta/llama-3.1-70b-instruct
Agent B: nvidia:qwen/qwen2.5-72b-instruct
Max Turns: 30
```

**After:**
```
🎯 Quick Actions
▶️  Start
⏹️  Stop
📊 Analyze

⚙️ Config
A: llama-3.1-70b-instruct
B: mixtral-8x7b-instruct
Turns: 30
```

**Footer:**
- Before: `n=New | s=Stop | a=Analyze | v=Archive | q=Quit`
- After: `n=Start | s=Stop | a=Analyze | q=Quit`

---

## 💡 Best Practices

### For Quick Tests
1. Use `test_templates.py quick`
2. Check results in web viewer
3. Iterate quickly

### For Deep Research
1. Use `test_templates.py extended`
2. Export JSON reports
3. Analyze in Python/R/Excel
4. Compare across conditions

### For Live Exploration
1. Use `python main.py` (TUI)
2. Watch conversations in real-time
3. Analyze interesting patterns
4. Save notable conversations

---

## 🔧 Troubleshooting

**Web viewer shows no conversations:**
```bash
# Check database
sqlite3 storage/conversations.db "SELECT COUNT(*) FROM conversations;"

# Run a test first
python test_templates.py quick
```

**TUI not scrolling:**
- Use mouse/trackpad
- Try arrow keys
- Press `j` (down) or `k` (up)

**Model errors:**
- Check `.env` has correct model names with `nvidia:` prefix
- Use `python test_nvidia.py` to verify API connection
- See `CUSTOM_MODELS.md` for valid model names

---

## 📚 Documentation

- `README.md` - Project overview
- `QUICKSTART.md` - 5-minute setup
- `QUANTITATIVE_ANALYSIS.md` - This file
- `CUSTOM_MODELS.md` - Model configuration
- `PROJECT_SUMMARY.md` - Technical details

---

## 🎉 Quick Commands Cheat Sheet

```bash
# Run conversations
python main.py                           # TUI
python test_templates.py quick           # Quick test
python test_templates.py list            # List templates

# View results
python viewer.py                         # Web viewer (http://localhost:5000)
python export_conversation.py 1 html     # Export to HTML

# Check database
sqlite3 storage/conversations.db "SELECT * FROM conversations;"
```

---

Happy researching! 🔬✨
