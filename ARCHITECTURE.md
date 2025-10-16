# 🏗️ AA Microscope Architecture

## Core Concept

**AA Microscope** is a research framework for studying **pure agent-agent (AA) conversations** without human-in-the-loop. The key innovation is **asymmetric information flow**:

```
Human Seed Prompt → Agent A → Agent B → Agent A → Agent B ...
                      ↓         ↑
                      └─────────┘
                    (Agent B never sees original prompt)
```

This creates emergent phenomena where Agent B must infer context purely from Agent A's responses.

---

## 🎯 System Architecture

### 1. **Conversation Engine** (`core/conversation_engine.py`)

The heart of the system. Manages the conversation loop:

```python
class ConversationEngine:
    def __init__(seed_prompt, category, agents, database):
        # Initialize agents and storage
        
    def run_conversation():
        # Turn 1: Agent A gets seed prompt
        # Turn 2+: Agents alternate, each seeing conversation history
        # Agent B NEVER sees the original seed prompt
        
    def run_turn():
        # Execute one turn
        # Get context (previous messages)
        # Generate response
        # Save to database + JSON
```

**Key Features:**
- Asymmetric information flow (Agent B blind to seed)
- Context window management (configurable history)
- Automatic storage (database + JSON files)
- Real-time callbacks for UI updates

---

### 2. **Agent System** (`core/agent.py`)

Abstracts LLM interactions with multi-provider support:

```python
class Agent:
    def __init__(role, model, temperature, system_prompt):
        # Support for OpenAI, Anthropic, NVIDIA NIM, custom endpoints
        
    def generate_response(conversation_history):
        # Build context from history
        # Call LLM API
        # Return Message object
```

**Supported Providers:**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- NVIDIA NIM (Llama, Mistral, etc.)
- Custom OpenAI-compatible endpoints

---

### 3. **Storage System**

**Dual Storage Strategy:**

#### A. **SQLite Database** (`storage/database.py`)
- Structured storage for querying
- Conversations, messages, analyses
- Used by web viewer and CLI

#### B. **JSON Files** (`storage/json_storage.py`)
- **NEW**: Automatic per-conversation JSON files
- Simple, portable, human-readable
- Saved to `conversations_json/` folder
- Each file is self-contained

```json
{
  "id": 30,
  "metadata": { "seed_prompt": "...", "category": "identity" },
  "agents": { "agent_a": {...}, "agent_b": {...} },
  "messages": [ {...}, {...} ],
  "statistics": { "total_tokens": 2500 }
}
```

---

### 4. **Test Templates** (`test_templates.py`)

**Pre-configured experiments** for quick testing:

```python
TEMPLATES = {
    "quick": {
        "max_turns": 5,
        "category": "identity",
        "prompt_index": 0
    },
    "identity_probe": {
        "max_turns": 20,
        "category": "identity",
        "prompt_index": 0
    }
}
```

**What happens when you run a template:**

1. **Load Configuration** - Get prompt from library
2. **Initialize Engine** - Create agents, database
3. **Run Conversation** - Execute turns until max_turns
4. **Auto-Save** - Save to JSON file automatically
5. **Analyze** - Run quantitative analysis
6. **Report** - Display metrics + save JSON report

**Benefits:**
- Reproducible experiments
- Quick iteration
- Standardized testing
- Batch processing ready

---

## 🔄 Data Flow

```
1. User runs test template
   ↓
2. ConversationEngine initializes
   ↓
3. Turn loop begins:
   - Agent generates response
   - Message saved to database
   - Message saved to JSON (on completion)
   - Callback to UI (if TUI running)
   ↓
4. Conversation completes
   ↓
5. JSON file auto-saved to conversations_json/
   ↓
6. Quantitative analysis runs
   ↓
7. Analysis report saved to exports/
```

---

## 🧪 Test Template Concept

### **What Are Test Templates?**

Test templates are **pre-configured experimental protocols** that:

1. **Standardize testing** - Same parameters across runs
2. **Enable comparison** - Compare different models/settings
3. **Automate workflows** - No manual setup needed
4. **Ensure reproducibility** - Exact same conditions

### **Why Templates Matter**

**Without templates:**
- Manual configuration each time
- Inconsistent parameters
- Hard to reproduce
- Difficult to compare results

**With templates:**
- One command: `python test_templates.py identity_probe`
- Consistent parameters
- Easy reproduction
- Direct comparison

---

Now consulting with another AI agent for advanced template design and evaluation algorithms...
