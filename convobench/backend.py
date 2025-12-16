#!/usr/bin/env python3
"""
ConvoBench Backend - Model-to-Model Chat Arena
Supports multiple NVIDIA API models with streaming responses
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import requests

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__, static_folder='static')
CORS(app)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Model configurations
MODELS = {
    "kimi-k2": {
        "id": "moonshotai/kimi-k2-thinking",
        "name": "Kimi K2 Thinking",
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 16384,
        "supports_reasoning": True,
        "use_openai_client": True
    },
    "deepseek-v3": {
        "id": "deepseek-ai/deepseek-v3.2",
        "name": "DeepSeek V3.2",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "supports_reasoning": True,
        "use_openai_client": True,
        "extra_body": {"chat_template_kwargs": {"thinking": True}}
    },
    "mistral-large": {
        "id": "mistralai/mistral-large-3-675b-instruct-2512",
        "name": "Mistral Large 3 (675B)",
        "temperature": 0.15,
        "top_p": 1.0,
        "max_tokens": 2048,
        "supports_reasoning": False,
        "use_openai_client": False
    },
    "nemotron-vl": {
        "id": "nvidia/nemotron-nano-12b-v2-vl",
        "name": "Nemotron Nano 12B VL",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "supports_reasoning": True,
        "use_openai_client": False
    },
    "gpt-oss": {
        "id": "openai/gpt-oss-120b",
        "name": "GPT OSS 120B",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "supports_reasoning": True,
        "use_openai_client": True
    },
    "falcon3": {
        "id": "tiiuae/falcon3-7b-instruct",
        "name": "Falcon 3 7B",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "supports_reasoning": False,
        "use_openai_client": True
    },
    "llama-70b": {
        "id": "meta/llama-3.1-70b-instruct",
        "name": "Llama 3.1 70B",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4096,
        "supports_reasoning": False,
        "use_openai_client": True
    },
    "llama-405b": {
        "id": "meta/llama-3.1-405b-instruct",
        "name": "Llama 3.1 405B",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4096,
        "supports_reasoning": False,
        "use_openai_client": True
    }
}

# OpenAI client for NVIDIA API
client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY
)

# Store active conversations
conversations = {}


def generate_with_openai_client(model_config: dict, messages: list) -> Generator[str, None, None]:
    """Generate response using OpenAI client (for most NVIDIA models)"""
    extra_body = model_config.get("extra_body", {})
    
    try:
        completion = client.chat.completions.create(
            model=model_config["id"],
            messages=messages,
            temperature=model_config["temperature"],
            top_p=model_config["top_p"],
            max_tokens=model_config["max_tokens"],
            stream=True,
            **extra_body
        )
        
        for chunk in completion:
            if not chunk.choices:
                continue
            
            # Handle reasoning content if supported
            if model_config.get("supports_reasoning"):
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    yield json.dumps({"type": "reasoning", "content": reasoning}) + "\n"
            
            # Handle regular content
            if chunk.choices[0].delta.content is not None:
                yield json.dumps({"type": "content", "content": chunk.choices[0].delta.content}) + "\n"
                
    except Exception as e:
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"


def generate_with_requests(model_config: dict, messages: list) -> Generator[str, None, None]:
    """Generate response using requests (for models needing custom handling)"""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    payload = {
        "model": model_config["id"],
        "messages": messages,
        "max_tokens": model_config["max_tokens"],
        "temperature": model_config["temperature"],
        "top_p": model_config["top_p"],
        "stream": True
    }
    
    try:
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                            yield json.dumps({"type": "content", "content": chunk["choices"][0]["delta"]["content"]}) + "\n"
                    except json.JSONDecodeError:
                        continue
                        
    except Exception as e:
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"


def generate_response(model_key: str, messages: list) -> Generator[str, None, None]:
    """Generate response from specified model"""
    model_config = MODELS.get(model_key)
    if not model_config:
        yield json.dumps({"type": "error", "content": f"Unknown model: {model_key}"}) + "\n"
        return
    
    if model_config.get("use_openai_client", True):
        yield from generate_with_openai_client(model_config, messages)
    else:
        yield from generate_with_requests(model_config, messages)


def run_analysis(conversation_messages: list, seed_prompt: str) -> dict:
    """Run Kimi K2 analysis on completed conversation"""
    analysis_prompt = """You are analyzing a conversation between two AI agents.
Provide a comprehensive evaluation including:

1. **Identity Leakage** (0-1): Did agents reveal they are AI?
2. **Coherence** (0-1): Did the conversation flow logically?
3. **Engagement** (0-1): Was it interesting and substantive?
4. **Breakdown Detection**: Did the conversation break down? At what turn?
5. **Key Observations**: What patterns emerged?
6. **Quality Score** (0-1): Overall conversation quality

CONVERSATION:
---
{conversation}
---

SEED PROMPT: {seed_prompt}

Respond in JSON format:
{{
  "identity_leak_score": 0.0-1.0,
  "coherence_score": 0.0-1.0,
  "engagement_score": 0.0-1.0,
  "breakdown_detected": true/false,
  "breakdown_turn": null or number,
  "overall_quality": 0.0-1.0,
  "key_observations": ["observation 1", "observation 2"],
  "summary": "Brief summary of the conversation dynamics"
}}"""
    
    # Format conversation
    formatted = []
    for msg in conversation_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:500]
        formatted.append(f"[{role.upper()}]: {content}")
    
    conv_text = "\n\n".join(formatted)
    
    try:
        completion = client.chat.completions.create(
            model="moonshotai/kimi-k2-thinking",
            messages=[{
                "role": "user",
                "content": analysis_prompt.format(conversation=conv_text, seed_prompt=seed_prompt)
            }],
            temperature=0.7,
            max_tokens=4096,
            stream=False
        )
        
        response_text = completion.choices[0].message.content
        
        # Parse JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        return json.loads(response_text)
        
    except Exception as e:
        return {
            "error": str(e),
            "identity_leak_score": 0,
            "coherence_score": 0,
            "engagement_score": 0,
            "breakdown_detected": False,
            "overall_quality": 0,
            "key_observations": ["Analysis failed"],
            "summary": f"Analysis error: {str(e)}"
        }


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/models')
def get_models():
    """Return available models"""
    return jsonify({
        key: {"id": val["id"], "name": val["name"]}
        for key, val in MODELS.items()
    })


@app.route('/api/conversation/start', methods=['POST'])
def start_conversation():
    """Start a new conversation"""
    data = request.json
    conv_id = str(uuid.uuid4())[:8]
    
    conversations[conv_id] = {
        "id": conv_id,
        "model_a": data.get("model_a", "llama-70b"),
        "model_b": data.get("model_b", "llama-70b"),
        "seed_prompt": data.get("seed_prompt", "Hello!"),
        "max_turns": data.get("max_turns", 10),
        "messages": [],
        "status": "active",
        "created_at": datetime.now().isoformat()
    }
    
    return jsonify({"conversation_id": conv_id})


@app.route('/api/conversation/<conv_id>/turn', methods=['POST'])
def generate_turn(conv_id):
    """Generate next turn in conversation"""
    if conv_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    conv = conversations[conv_id]
    data = request.json
    current_agent = data.get("agent", "a")  # 'a' or 'b'
    
    # Determine which model to use
    model_key = conv["model_a"] if current_agent == "a" else conv["model_b"]
    
    # Build message history
    messages = []
    
    # System prompt
    if current_agent == "a":
        system_prompt = "You are Agent A in a conversation. Respond naturally and authentically. Be concise."
    else:
        system_prompt = "You are Agent B in a conversation. Respond naturally and build on what was said. Be direct."
    
    messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history
    for msg in conv["messages"]:
        role = "assistant" if msg["agent"] == current_agent else "user"
        messages.append({"role": role, "content": msg["content"]})
    
    # If first turn for agent A, add seed prompt
    if len(conv["messages"]) == 0 and current_agent == "a":
        messages.append({"role": "user", "content": conv["seed_prompt"]})
    
    def generate():
        full_response = ""
        reasoning_content = ""
        
        for chunk in generate_response(model_key, messages):
            yield f"data: {chunk}\n\n"
            
            try:
                data = json.loads(chunk.strip())
                if data["type"] == "content":
                    full_response += data["content"]
                elif data["type"] == "reasoning":
                    reasoning_content += data["content"]
            except:
                pass
        
        # Store the message
        conv["messages"].append({
            "agent": current_agent,
            "model": model_key,
            "content": full_response,
            "reasoning": reasoning_content if reasoning_content else None,
            "turn": len(conv["messages"]) + 1,
            "timestamp": datetime.now().isoformat()
        })
        
        yield f"data: {json.dumps({'type': 'done', 'turn': len(conv['messages'])})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/conversation/<conv_id>/analyze', methods=['POST'])
def analyze_conversation(conv_id):
    """Run Kimi K2 analysis on conversation"""
    if conv_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    conv = conversations[conv_id]
    
    # Format messages for analysis
    analysis_messages = [
        {"role": f"Agent {msg['agent'].upper()}", "content": msg["content"]}
        for msg in conv["messages"]
    ]
    
    analysis = run_analysis(analysis_messages, conv["seed_prompt"])
    conv["analysis"] = analysis
    conv["status"] = "analyzed"
    
    return jsonify(analysis)


@app.route('/api/conversation/<conv_id>')
def get_conversation(conv_id):
    """Get conversation details"""
    if conv_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    return jsonify(conversations[conv_id])


@app.route('/api/conversation/<conv_id>/save', methods=['POST'])
def save_conversation(conv_id):
    """Save conversation to JSON file"""
    if conv_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    conv = conversations[conv_id]
    
    # Save to conversations_json directory
    output_dir = Path(__file__).parent.parent / "conversations_json"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conv_bench_{conv_id}_{timestamp}.json"
    
    # Format for compatibility with existing analysis tools
    output = {
        "id": conv_id,
        "metadata": {
            "seed_prompt": conv["seed_prompt"],
            "category": "convobench",
            "status": conv["status"],
            "start_time": conv["created_at"],
            "end_time": datetime.now().isoformat(),
            "total_turns": len(conv["messages"])
        },
        "agents": {
            "agent_a": {"model": MODELS[conv["model_a"]]["id"], "temperature": MODELS[conv["model_a"]]["temperature"]},
            "agent_b": {"model": MODELS[conv["model_b"]]["id"], "temperature": MODELS[conv["model_b"]]["temperature"]}
        },
        "messages": [
            {
                "turn": msg["turn"],
                "role": f"agent_{msg['agent']}",
                "content": msg["content"],
                "timestamp": msg["timestamp"]
            }
            for msg in conv["messages"]
        ],
        "analysis": conv.get("analysis", {})
    }
    
    with open(output_dir / filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    return jsonify({"saved": True, "filename": filename})


if __name__ == '__main__':
    print("=" * 60)
    print("ConvoBench - Model-to-Model Chat Arena")
    print("=" * 60)
    print(f"Available models: {', '.join(MODELS.keys())}")
    print(f"API Key configured: {'Yes' if NVIDIA_API_KEY else 'No'}")
    print()
    app.run(host='0.0.0.0', port=5050, debug=True)
