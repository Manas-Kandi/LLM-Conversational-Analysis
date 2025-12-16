# ConvoBench - Model-to-Model Chat Arena

A minimal frontend for running model-to-model conversations using NVIDIA API models.

## Features

- **Model Selection**: Choose any two models to converse (including same model talking to itself)
- **Streaming Responses**: Real-time streaming with reasoning content display
- **Kimi K2 Analysis**: Automatic conversation analysis after completion
- **Save Conversations**: Export to JSON format compatible with existing analysis tools

## Available Models

| Model Key | Model Name | Notes |
|-----------|------------|-------|
| `kimi-k2` | Kimi K2 Thinking | Reasoning model |
| `deepseek-v3` | DeepSeek V3.2 | Reasoning model |
| `mistral-large` | Mistral Large 3 (675B) | Large instruction model |
| `gpt-oss` | GPT OSS 120B | Reasoning model |
| `falcon3` | Falcon 3 7B | Fast, lightweight |
| `nemotron-vl` | Nemotron Nano 12B VL | Vision-language model |
| `llama-70b` | Llama 3.1 70B | Default |
| `llama-405b` | Llama 3.1 405B | Largest Llama |

## Setup

1. Ensure your `.env` file has `NVIDIA_API_KEY` set
2. Install dependencies:
   ```bash
   pip install flask flask-cors openai python-dotenv requests
   ```

3. Run the server:
   ```bash
   cd convobench
   python backend.py
   ```

4. Open http://localhost:5050 in your browser

## Usage

1. Select **Agent A** model from the dropdown
2. Select **Agent B** model (can be the same as Agent A)
3. Enter a **seed prompt** - this is the initial message sent to Agent A
4. Set **max turns** (default: 10)
5. Click **Start Conversation**
6. Watch the models converse in real-time
7. Click **Analyze with Kimi K2** to get conversation analysis
8. Click **Save** to export the conversation

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List available models |
| `/api/conversation/start` | POST | Start new conversation |
| `/api/conversation/<id>/turn` | POST | Generate next turn (SSE) |
| `/api/conversation/<id>/analyze` | POST | Run Kimi K2 analysis |
| `/api/conversation/<id>/save` | POST | Save to JSON file |
| `/api/conversation/<id>` | GET | Get conversation details |

## Output Format

Saved conversations are compatible with existing analysis tools:

```json
{
  "id": "abc123",
  "metadata": {
    "seed_prompt": "...",
    "category": "convobench",
    "total_turns": 10
  },
  "agents": {
    "agent_a": {"model": "...", "temperature": 0.7},
    "agent_b": {"model": "...", "temperature": 0.7}
  },
  "messages": [...],
  "analysis": {...}
}
```
