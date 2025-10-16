# ⚡ Quick Start - AA Microscope

Get running in 5 minutes!

---

## 1️⃣ Install Dependencies (1 min)

```bash
cd /Users/manaskandimalla/Desktop/Projects/pekoflabs
pip install -r requirements.txt
```

---

## 2️⃣ Configure API Keys (1 min)

```bash
cp .env.example .env
nano .env  # or your preferred editor
```

Add your API key:
```env
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here
# OR use NVIDIA NIM (Llama, Qwen, Mistral, etc.)
NVIDIA_API_KEY=nvapi-your-key-here
# OR use custom OpenAI-compatible endpoints
CUSTOM_API_KEY_1=your-custom-key
CUSTOM_BASE_URL_1=https://your-endpoint.com/v1
```

Save and exit.

**Using NVIDIA or custom models?** See `CUSTOM_MODELS.md` for details!

---

## 3️⃣ Launch! (30 seconds)

### Option A: Beautiful Terminal UI (Recommended)
```bash
python main.py
```

- Press `n` or click "New Conversation"
- Select a prompt from the library
- Watch agents converse in real-time!
- Press `a` to analyze when done

### Option B: Simple Example Script
```bash
python examples/quick_start.py
```

### Option C: Command Line
```bash
# List available prompts
python cli.py prompts

# Run a conversation
python cli.py run --category identity --index 0 --max-turns 10
```

---

## 📊 Your First Analysis

After a conversation completes:

```bash
python cli.py analyze 1
```

This runs all analyses:
- ✅ Statistical metrics
- ✅ Semantic drift tracking
- ✅ Role detection
- ✅ Pattern recognition

---

## 📁 Export Results

```bash
# Export as Markdown (great for reading)
python cli.py export 1 --format markdown

# Export as JSON (for further processing)
python cli.py export 1 --format json

# Full analysis report
python cli.py analyze 1 --output my_analysis.md
```

---

## 🎯 Key Commands

| Action | Command |
|--------|---------|
| Launch TUI | `python main.py` |
| List prompts | `python cli.py prompts` |
| Run conversation | `python cli.py run --category <cat> --index <i>` |
| Analyze | `python cli.py analyze <id>` |
| Export | `python cli.py export <id>` |
| View archive | `python cli.py list` |
| Statistics | `python cli.py stats` |

---

## 🔬 What to Try First

1. **Identity Confusion Prompt**
   - Watch for AI self-awareness moments
   - Category: `identity`, Index: `0`

2. **Philosophical Discussion**
   - High semantic drift
   - Category: `meta_cognition`, Index: `0`

3. **Emotional Exchange**
   - Empathy cascade effects
   - Category: `emotional`, Index: `0`

4. **Pure Chaos**
   - Stress test with nonsense
   - Category: `chaos`, Index: `0`

---

## 🤖 Using Other Models

**NVIDIA NIM (Free/Cheap):**
```bash
AGENT_A_MODEL=nvidia:qwen/qwen3-next-80b-a3b-instruct
AGENT_B_MODEL=nvidia:meta/llama-3.1-70b-instruct
```

**Mix Models:**
```bash
AGENT_A_MODEL=gpt-4
AGENT_B_MODEL=nvidia:meta/llama-3.1-70b-instruct
```

**Run example:**
```bash
python examples/nvidia_models_example.py
```

**Full guide:** See `CUSTOM_MODELS.md`

---

## 💡 Pro Tips

- **Start with 10-15 turns** for quick experiments
- **Use NVIDIA NIM models** for free/cheap testing
- **Run statistical analysis first** (it's instant)
- **Save interesting conversations** immediately
- **Compare similar prompts** with different temperatures
- **Mix model providers** to study cross-model dynamics

---

## 🆘 Problems?

**"Configuration Error"**
→ Check `.env` file has valid API key

**"Module not found"**
→ Run: `pip install -r requirements.txt`

**"API Error"**
→ Verify your API key is active and has credits

**More help:**
→ See `SETUP_GUIDE.md` for detailed troubleshooting

---

## 🎉 You're All Set!

Start exploring emergent phenomena in agent-agent dialogue:

```bash
python main.py
```

**For detailed documentation:** See `README.md` and `SETUP_GUIDE.md`

---

**Happy researching! 🔬✨**
