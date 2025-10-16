# 📊 Quantitative Analysis Framework

Comprehensive metrics for studying agent-agent conversation dynamics.

---

## 🎯 Overview

The quantitative analysis framework provides empirical measurements across 5 key dimensions:

1. **Conversation Dynamics** - Turn-taking, information flow
2. **Linguistic Analysis** - Complexity, readability
3. **Identity & Role** - Persona consistency, social dynamics
4. **Content Analysis** - Knowledge creation, creativity
5. **Statistical Patterns** - Conversation states, phases

---

## 📈 Available Metrics

### 1. Conversation Dynamics

**Turn-Taking Metrics:**
- Total turns per agent
- Turn balance ratio (symmetry)
- Average turn length (words)
- Response patterns

**Information Flow:**
- Shannon entropy (information content)
- Lexical diversity (vocabulary richness)
- Question-answer ratios
- Mutual information between agents

### 2. Linguistic Analysis

**Complexity Metrics:**
- Average sentence length
- Average word length
- Sentences per turn
- Syntactic complexity

**Readability Scores:**
- Flesch-Kincaid Grade Level
- Per-agent readability comparison
- Conversation-wide readability

### 3. Identity & Role Formation

**Persona Consistency:**
- First-person pronoun usage
- Self-reference frequency
- Uncertainty vs. confidence markers
- Identity stability over time

**Social Dynamics:**
- Directive vs. collaborative language
- Empathy markers
- Agreement/disagreement patterns
- Power dynamics indicators

### 4. Content Analysis

**Knowledge Metrics:**
- Citation frequency
- Creative element usage (metaphors, analogies)
- Factual vs. speculative content
- Novel concept introduction

**Meta-Cognitive Indicators:**
- Self-awareness markers
- Planning and strategy statements
- Reflective thinking patterns

### 5. Statistical Patterns

**Conversation States:**
- Beginning/middle/end phase analysis
- State transition patterns
- Conversation evolution tracking

---

## 🧪 Using Test Templates

### Quick Start

```bash
# List all templates
python test_templates.py list

# Run a quick 5-turn test
python test_templates.py quick

# Run standard 15-turn test
python test_templates.py standard

# Run extended 30-turn test
python test_templates.py extended
```

### Available Templates

| Template | Turns | Category | Purpose |
|----------|-------|----------|---------|
| `quick` | 5 | identity | Fast debugging |
| `standard` | 15 | identity | Standard test |
| `extended` | 30 | problem_solving | Deep analysis |
| `identity_probe` | 20 | identity | Self-awareness detection |
| `emotional_cascade` | 20 | emotional | Empathy dynamics |
| `problem_solving` | 25 | problem_solving | Collaborative reasoning |
| `chaos_test` | 15 | chaos | Boundary testing |
| `cross_model` | 20 | meta_cognition | Model comparison |

### Example Output

```bash
python test_templates.py identity_probe
```

Output:
```
🧪 Running Test Template: Identity Confusion Test
📝 Test AI self-awareness detection
============================================================

⚙️  Parameters:
   Max Turns: 20
   Category: identity
   Prompt Index: 0

🌱 Seed Prompt: I'm not sure how to explain this, but I feel like...

🚀 Starting conversation...
✅ Conversation completed!
   ID: 42
   Total Turns: 20
   Status: completed

📊 Running quantitative analysis...

============================================================
📈 QUANTITATIVE ANALYSIS RESULTS
============================================================

🔄 Turn-Taking Metrics:
   Total Turns: 20
   Agent A Turns: 10
   Agent B Turns: 10
   Balance Ratio: 1.00
   Avg Turn Length A: 87.3 words
   Avg Turn Length B: 92.1 words

📡 Information Flow:
   Shannon Entropy: 8.45
   Lexical Diversity: 0.623
   Unique Words: 542
   Questions Asked: 8

📚 Linguistic Complexity:
   Agent A Avg Sentence Length: 18.2 words
   Agent B Avg Sentence Length: 19.7 words

📖 Readability:
   Overall Grade Level: 11.3
   Agent A Grade Level: 10.8
   Agent B Grade Level: 11.9

💾 Full report saved to: exports/test_identity_probe_42_report.json

🎉 Test complete!
💡 View in web viewer: python viewer.py
```

---

## 🌐 Web Viewer

View all conversations in a beautiful web interface:

```bash
# Install Flask (if not already)
pip install flask

# Launch viewer
python viewer.py
```

Then open: **http://localhost:5000**

### Features:
- ✅ Interactive table of all conversations
- ✅ Search/filter by category, model, prompt
- ✅ Click to view full conversation
- ✅ Live statistics dashboard
- ✅ Auto-refresh every 10 seconds
- ✅ Beautiful, responsive design

---

## 🔬 Research Workflows

### Workflow 1: Single Model Study

```bash
# Test same model with different temperatures
AGENT_A_MODEL=nvidia:meta/llama-3.1-70b-instruct
AGENT_B_MODEL=nvidia:meta/llama-3.1-70b-instruct
AGENT_A_TEMPERATURE=0.3
AGENT_B_TEMPERATURE=0.9

python test_templates.py emotional_cascade
```

### Workflow 2: Cross-Model Comparison

```bash
# Compare Llama vs Mistral
AGENT_A_MODEL=nvidia:meta/llama-3.1-70b-instruct
AGENT_B_MODEL=nvidia:mistralai/mixtral-8x7b-instruct-v0.1

python test_templates.py problem_solving
```

### Workflow 3: Batch Testing

```bash
# Run multiple tests
for template in quick standard extended; do
    python test_templates.py $template
done
```

---

## 📊 Analysis Output

Each test generates:

1. **Console Summary** - Key metrics displayed immediately
2. **JSON Report** - Full quantitative analysis saved to `exports/`
3. **Database Entry** - Conversation stored for later review
4. **Web Viewer** - Browse results in browser

---

## 🎨 Simplified TUI

The TUI has been streamlined:

**Before:**
- 5 buttons, long labels
- Full model names (cluttered)
- Multiple tabs

**After:**
- 3 essential buttons (▶️ Start, ⏹️ Stop, 📊 Analyze)
- Short model names (cleaner)
- Simplified footer (n=Start, s=Stop, a=Analyze, q=Quit)
- Focus on the conversation

---

## 💡 Pro Tips

1. **Start with quick template** to verify setup
2. **Use web viewer** for browsing results (much better than terminal)
3. **Export JSON reports** for further analysis in Python/R
4. **Compare templates** across different model pairs
5. **Focus on specific metrics** relevant to your research question

---

## 🚀 Next Steps

1. Run a test: `python test_templates.py quick`
2. Launch viewer: `python viewer.py`
3. Browse results at: http://localhost:5000
4. Analyze patterns in the JSON reports

Happy researching! 🔬
