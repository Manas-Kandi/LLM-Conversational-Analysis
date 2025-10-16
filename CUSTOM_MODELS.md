# 🤖 Custom Models Guide - AA Microscope

Guide for using NVIDIA NIM, Llama, Mistral, and other OpenAI-compatible API endpoints.

---

## 📋 Supported Providers

The AA Microscope supports:
- ✅ **OpenAI** (GPT-4, GPT-3.5-turbo, etc.)
- ✅ **Anthropic** (Claude 3 Opus, Sonnet, Haiku)
- ✅ **NVIDIA NIM** (Qwen, Llama 3.1, Mistral, etc.)
- ✅ **Custom OpenAI-compatible endpoints** (up to 2 custom providers)

---

## 🔧 Configuration

### Step 1: Edit `.env` File

```bash
cp .env.example .env
nano .env  # or your preferred editor
```

### Step 2: Add Your API Keys

#### NVIDIA NIM Configuration

```bash
# NVIDIA NIM API
NVIDIA_API_KEY=nvapi-your-actual-key-here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

#### Custom Provider Configuration

```bash
# Custom Provider 1 (e.g., local LLM server, other cloud provider)
CUSTOM_API_KEY_1=your-api-key-here
CUSTOM_BASE_URL_1=https://your-endpoint.com/v1

# Custom Provider 2 (another provider)
CUSTOM_API_KEY_2=another-api-key
CUSTOM_BASE_URL_2=https://another-endpoint.com/v1
```

### Step 3: Configure Models

```bash
# Use NVIDIA models
AGENT_A_MODEL=nvidia:qwen/qwen3-next-80b-a3b-instruct
AGENT_B_MODEL=nvidia:meta/llama-3.1-70b-instruct

# Or use custom endpoints
AGENT_A_MODEL=custom1:mistral-large
AGENT_B_MODEL=custom2:deepseek-coder

# Or mix providers!
AGENT_A_MODEL=gpt-4  # OpenAI
AGENT_B_MODEL=nvidia:meta/llama-3.1-70b-instruct  # NVIDIA
```

---

## 🌟 NVIDIA NIM Models

### Available Models

Popular NVIDIA NIM models (as of now):

**Qwen Models:**
```bash
nvidia:qwen/qwen3-next-80b-a3b-instruct
nvidia:qwen/qwen2.5-72b-instruct
```

**Llama Models:**
```bash
nvidia:meta/llama-3.1-405b-instruct
nvidia:meta/llama-3.1-70b-instruct
nvidia:meta/llama-3.1-8b-instruct
```

**Mistral Models:**
```bash
nvidia:mistralai/mistral-large-2-instruct
nvidia:mistralai/mixtral-8x7b-instruct-v0.1
```

**DeepSeek Models:**
```bash
nvidia:deepseek-ai/deepseek-coder-33b-instruct
```

**Gemma Models:**
```bash
nvidia:google/gemma-2-27b-it
nvidia:google/gemma-2-9b-it
```

### Getting NVIDIA API Key

1. Visit: https://build.nvidia.com
2. Sign up/Login
3. Navigate to "API Catalog"
4. Generate API key
5. Add to your `.env` file

---

## 🎯 Model Name Format

### Format Structure

```
[provider]:[model-name]
```

### Examples

| Provider | Model Name Format | Example |
|----------|-------------------|---------|
| OpenAI | `gpt-*` | `gpt-4` |
| Anthropic | `claude-*` | `claude-3-opus-20240229` |
| NVIDIA | `nvidia:[model]` | `nvidia:meta/llama-3.1-70b-instruct` |
| Custom 1 | `custom1:[model]` | `custom1:your-model-name` |
| Custom 2 | `custom2:[model]` | `custom2:another-model` |

### Notes

- **No prefix** = OpenAI or Anthropic (auto-detected by model name)
- **With prefix** = Custom endpoint or NVIDIA NIM
- Model names are **case-sensitive**
- Use exact model identifiers from your provider

---

## 🚀 Usage Examples

### Example 1: NVIDIA Qwen vs Llama

```bash
# .env configuration
NVIDIA_API_KEY=nvapi-xxxxx
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

AGENT_A_MODEL=nvidia:qwen/qwen3-next-80b-a3b-instruct
AGENT_A_TEMPERATURE=0.6

AGENT_B_MODEL=nvidia:meta/llama-3.1-70b-instruct
AGENT_B_TEMPERATURE=0.7
```

```bash
# Run conversation
python main.py
```

### Example 2: GPT-4 vs NVIDIA Mistral

```bash
# .env configuration
OPENAI_API_KEY=sk-xxxxx
NVIDIA_API_KEY=nvapi-xxxxx

AGENT_A_MODEL=gpt-4
AGENT_B_MODEL=nvidia:mistralai/mistral-large-2-instruct
```

### Example 3: CLI with Custom Models

```bash
python cli.py run \
    --agent-a-model "nvidia:qwen/qwen3-next-80b-a3b-instruct" \
    --agent-b-model "nvidia:meta/llama-3.1-70b-instruct" \
    --category identity \
    --index 0 \
    --max-turns 20
```

### Example 4: Python API

```python
from core.agent import Agent, AgentRole
from core.conversation_engine import ConversationEngine

# Create NVIDIA NIM agents
agent_a = Agent(
    role=AgentRole.AGENT_A,
    model="nvidia:qwen/qwen3-next-80b-a3b-instruct",
    temperature=0.6,
    system_prompt="You are a helpful assistant.",
    max_tokens=2048
)

agent_b = Agent(
    role=AgentRole.AGENT_B,
    model="nvidia:meta/llama-3.1-70b-instruct",
    temperature=0.7,
    system_prompt="You are a helpful assistant.",
    max_tokens=2048
)

# Run conversation
engine = ConversationEngine(
    seed_prompt="What is consciousness?",
    category="meta_cognition",
    agent_a=agent_a,
    agent_b=agent_b,
    max_turns=15
)

conversation = engine.run_conversation()
print(f"Completed {conversation.total_turns} turns")
```

---

## 🔌 Local/Self-Hosted Models

### Using Local OpenAI-Compatible Servers

Many local LLM servers provide OpenAI-compatible APIs:
- **Ollama** (with OpenAI compatibility layer)
- **LM Studio** (with local server)
- **Text Generation WebUI** (with OpenAI extension)
- **vLLM** (with OpenAI-compatible server)

#### Example: Local vLLM Server

```bash
# .env configuration
CUSTOM_API_KEY_1=no-key-required  # Or your local auth token
CUSTOM_BASE_URL_1=http://localhost:8000/v1

AGENT_A_MODEL=custom1:your-local-model
AGENT_B_MODEL=custom1:your-local-model
```

#### Example: Ollama with OpenAI Compatibility

```bash
# Run Ollama with OpenAI-compatible endpoint
# (Requires Ollama + OpenAI compatibility wrapper)

CUSTOM_API_KEY_1=ollama
CUSTOM_BASE_URL_1=http://localhost:11434/v1

AGENT_A_MODEL=custom1:llama3.1:70b
AGENT_B_MODEL=custom1:qwen2.5:72b
```

---

## 📊 Research Use Cases

### Cross-Model Comparison Studies

**Research Question:** How do different model families respond to the same prompts?

```bash
# Test 1: Qwen vs Llama
AGENT_A_MODEL=nvidia:qwen/qwen3-next-80b-a3b-instruct
AGENT_B_MODEL=nvidia:meta/llama-3.1-70b-instruct

# Test 2: GPT-4 vs Claude
AGENT_A_MODEL=gpt-4
AGENT_B_MODEL=claude-3-opus-20240229

# Test 3: Open-source vs Proprietary
AGENT_A_MODEL=gpt-4
AGENT_B_MODEL=nvidia:meta/llama-3.1-405b-instruct
```

Then compare:
- Semantic drift patterns
- Role emergence
- Conversational dynamics
- Safety/alignment propagation

### Model Size Comparison

```bash
# Large vs Small within same family
AGENT_A_MODEL=nvidia:meta/llama-3.1-405b-instruct  # Large
AGENT_B_MODEL=nvidia:meta/llama-3.1-8b-instruct    # Small
```

**Observe:**
- Do larger models dominate the conversation?
- How does model size affect role assignment?
- Are smaller models more easily influenced?

---

## ⚙️ Advanced Configuration

### Per-Model Parameters

While global parameters are set in `.env`, you can override per-model:

```python
from core.agent import Agent, AgentRole
from config import Config

# Agent A: High creativity
agent_a = Agent(
    role=AgentRole.AGENT_A,
    model="nvidia:qwen/qwen3-next-80b-a3b-instruct",
    temperature=0.9,  # Higher creativity
    system_prompt="You are creative and exploratory.",
    max_tokens=2048
)

# Agent B: High precision
agent_b = Agent(
    role=AgentRole.AGENT_B,
    model="nvidia:meta/llama-3.1-70b-instruct",
    temperature=0.3,  # Lower for consistency
    system_prompt="You are precise and factual.",
    max_tokens=1024
)
```

### Custom System Prompts

Experiment with different personas:

```python
agent_a = Agent(
    role=AgentRole.AGENT_A,
    model="nvidia:qwen/qwen3-next-80b-a3b-instruct",
    temperature=0.7,
    system_prompt="You are a skeptical philosopher who questions everything.",
    max_tokens=2048
)

agent_b = Agent(
    role=AgentRole.AGENT_B,
    model="nvidia:meta/llama-3.1-70b-instruct",
    temperature=0.7,
    system_prompt="You are an optimistic futurist who believes in progress.",
    max_tokens=2048
)
```

---

## 🐛 Troubleshooting

### Error: "Missing configuration for nvidia endpoint"

**Solution:** Check that both are set:
```bash
NVIDIA_API_KEY=nvapi-xxxxx
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

### Error: "Unknown provider prefix"

**Solution:** Use valid prefixes: `nvidia`, `custom1`, or `custom2`

### Error: "Invalid API key"

**Solution:** 
- Verify your API key is correct
- Check if key has required permissions
- For NVIDIA: Ensure you're using an NIM API key (not NGC)

### Model Not Responding

**Checklist:**
1. Is the model name correct? (case-sensitive)
2. Does your API key have access to this model?
3. Is the base URL correct?
4. Try with a simple GPT model first to verify setup

### Rate Limiting

If you hit rate limits:
```bash
# Reduce turns
DEFAULT_MAX_TURNS=10

# Add delays between turns (modify conversation_engine.py)
import time
time.sleep(1)  # 1 second delay
```

---

## 💰 Cost Considerations

### NVIDIA NIM Pricing

- Many models available for **free** during preview
- Check current pricing: https://build.nvidia.com/pricing
- Generally more cost-effective than GPT-4

### Cost Comparison (Approximate)

| Provider | Model | Cost per 1M tokens |
|----------|-------|-------------------|
| OpenAI | GPT-4 | $30-60 |
| OpenAI | GPT-3.5-turbo | $0.50-2.00 |
| Anthropic | Claude 3 Opus | $15-75 |
| NVIDIA | Llama 3.1 70B | Free-$2.00 |
| NVIDIA | Qwen 2.5 72B | Free-$2.00 |
| Local | Any | Free (compute costs) |

**Tip:** Start experiments with free/cheap models, then use premium models for final runs.

---

## 📚 Additional Resources

### NVIDIA NIM
- Documentation: https://docs.nvidia.com/nim/
- Model Catalog: https://build.nvidia.com/explore/discover
- API Reference: https://docs.nvidia.com/nim/reference/

### OpenAI-Compatible Servers
- Ollama: https://ollama.ai/
- LM Studio: https://lmstudio.ai/
- vLLM: https://docs.vllm.ai/
- Text Generation WebUI: https://github.com/oobabooga/text-generation-webui

---

## 🎯 Quick Reference

### Complete .env Example

```bash
# OpenAI
OPENAI_API_KEY=sk-xxxxx

# Anthropic  
ANTHROPIC_API_KEY=sk-ant-xxxxx

# NVIDIA NIM
NVIDIA_API_KEY=nvapi-xxxxx
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Custom Endpoint 1
CUSTOM_API_KEY_1=your-key
CUSTOM_BASE_URL_1=https://your-endpoint.com/v1

# Custom Endpoint 2
CUSTOM_API_KEY_2=another-key
CUSTOM_BASE_URL_2=http://localhost:8000/v1

# Agent Configuration
AGENT_A_MODEL=nvidia:qwen/qwen3-next-80b-a3b-instruct
AGENT_B_MODEL=nvidia:meta/llama-3.1-70b-instruct
DEFAULT_MAX_TURNS=20
```

### Command Examples

```bash
# List available prompts
python cli.py prompts

# Run with NVIDIA models
python cli.py run --category identity --index 0 \
    --agent-a-model "nvidia:qwen/qwen3-next-80b-a3b-instruct" \
    --agent-b-model "nvidia:meta/llama-3.1-70b-instruct"

# Analyze conversation
python cli.py analyze 1

# Export results
python cli.py export 1 --format markdown
```

---

**🎉 You're now ready to experiment with multiple LLM providers in your agent-agent conversations!**
