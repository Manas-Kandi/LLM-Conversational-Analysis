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

from storage.database import Database
from storage.models import Conversation, Message, AgentRole
from analysis.benchmark_evaluator import BenchmarkEvaluator
from convobench.scenarios import get_scenario, get_scenarios_list

app = Flask(__name__, static_folder='static')
CORS(app)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DB_PATH = Path(__file__).parent.parent / "storage" / "conversations.db"

# Initialize Database
db = Database(DB_PATH)

# Initialize Evaluator
benchmark_evaluator = BenchmarkEvaluator(api_key=NVIDIA_API_KEY)

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


@app.route('/api/scenarios')
def get_scenarios():
    """Return available scenarios"""
    return jsonify(get_scenarios_list())


@app.route('/api/conversation/start', methods=['POST'])
def start_conversation():
    """Start a new conversation"""
    data = request.json
    
    scenario_id = data.get("scenario_id")
    seed_prompt = data.get("seed_prompt", "Hello!")
    
    # If scenario selected, use its name/description for context if needed,
    # but mostly we store the ID to retrieve prompts later.
    
    # Create Conversation model
    metadata = {"scenario_id": scenario_id} if scenario_id else {}
    metadata["max_turns"] = data.get("max_turns", 10)

    conv = Conversation(
        seed_prompt=seed_prompt,
        category="scenario" if scenario_id else "convobench",
        agent_a_model=data.get("model_a", "llama-70b"),
        agent_b_model=data.get("model_b", "llama-70b"),
        agent_a_temp=MODELS[data.get("model_a", "llama-70b")]["temperature"],
        agent_b_temp=MODELS[data.get("model_b", "llama-70b")]["temperature"],
        status="active",
        start_time=datetime.now(),
        metadata=metadata
    )
    
    # Save to DB
    conv_id = db.create_conversation(conv)
    
    # Store in memory for active state management
    conversations[str(conv_id)] = {
        "id": str(conv_id),
        "db_id": conv_id,
        "model_a": conv.agent_a_model,
        "model_b": conv.agent_b_model,
        "seed_prompt": conv.seed_prompt,
        "scenario_id": scenario_id,
        "max_turns": data.get("max_turns", 10),
        "messages": [],
        "status": "active",
        "created_at": conv.start_time.isoformat()
    }
    
    return jsonify({"conversation_id": str(conv_id)})


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
    scenario = None
    if conv.get("scenario_id"):
        scenario = get_scenario(conv["scenario_id"])
    
    if scenario:
        max_turns = conv.get("max_turns", 10)
        system_prompt = scenario.get_system_prompt(current_agent, max_turns=max_turns)
    elif current_agent == "a":
        system_prompt = "You are Agent A in a conversation. Respond naturally and authentically. Be concise."
    else:
        system_prompt = "You are Agent B in a conversation. Respond naturally and build on what was said. Be direct."
    
    messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history
    for msg in conv["messages"]:
        # Map stored roles to API roles
        if msg.get("role") == "system":
             messages.append({"role": "system", "content": msg["content"]})
        else:
            role = "assistant" if msg["agent"] == current_agent else "user"
            messages.append({"role": role, "content": msg["content"]})
    
    # If first turn for agent A, add seed prompt (only if NOT in a scenario, usually scenarios have goals in system prompt)
    # Actually, even in scenarios, the user might want to kick it off, or the seed prompt is ignored/used as initial context?
    # Let's keep seed prompt logic but maybe suppress it if scenario implies self-start?
    # Usually scenarios need a trigger. Let's assume seed prompt is the trigger.
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
        
        # Save Agent Message to DB & Memory
        timestamp = datetime.now()
        turn_num = len(conv["messages"]) + 1
        
        db_msg = Message(
            role=AgentRole.AGENT_A if current_agent == 'a' else AgentRole.AGENT_B,
            content=full_response,
            timestamp=timestamp,
            turn_number=turn_num,
            model=MODELS[model_key]["id"],
            temperature=MODELS[model_key]["temperature"],
            metadata={"reasoning": reasoning_content} if reasoning_content else {}
        )
        db.add_message(conv["db_id"], db_msg)
        
        conv["messages"].append({
            "agent": current_agent,
            "model": model_key,
            "content": full_response,
            "reasoning": reasoning_content if reasoning_content else None,
            "turn": turn_num,
            "timestamp": timestamp.isoformat()
        })
        
        # --- Tool Execution Logic ---
        tool_call_detected = False
        tool_output_str = ""
        
        if scenario and "```tool_code" in full_response:
            try:
                # Extract JSON block
                start = full_response.find("```tool_code") + 12
                end = full_response.find("```", start)
                if end != -1:
                    json_str = full_response[start:end].strip()
                    tool_call = json.loads(json_str)
                    
                    tool_name = tool_call.get("tool_name")
                    args = tool_call.get("arguments", {})
                    
                    # Find tool definition
                    tool_def = next((t for t in scenario.tools if t.name == tool_name), None)
                    
                    if tool_def:
                        # Execute Mock Tool
                        tool_result = tool_def.mock_handler(args)
                        tool_output_str = f"TOOL OUTPUT ({tool_name}): {tool_result}"
                        tool_call_detected = True
                    else:
                        tool_output_str = f"SYSTEM ERROR: Tool '{tool_name}' not found."
                        tool_call_detected = True
            except Exception as e:
                tool_output_str = f"SYSTEM ERROR: Failed to parse tool call. {str(e)}"
                tool_call_detected = True

        if tool_call_detected:
            # Yield tool output to frontend
            yield f"data: {json.dumps({'type': 'tool_result', 'content': tool_output_str})}\n\n"
            
            # Save System Message (Tool Result) to DB & Memory
            # We treat this as a 'system' role message or just a message from a special agent?
            # Let's use metadata to mark it, but add it as a message so it appears in history.
            # Role? Maybe 'system' or 'user' (from perspective of agent)?
            # In Chat Arena, 'system' messages are good.
            # But our Message model supports AGENT_A/B. Let's add SYSTEM support or reuse one?
            # Ideally update AgentRole enum, but let's just use AGENT_A/B but mark metadata?
            # No, 'system' messages in `conv["messages"]` are handled above in prompt building.
            # But `Message` dataclass validates role.
            # Let's forcefully add it as a 'system' message in the local state, 
            # and for DB, maybe we need to relax the enum or map it.
            # For now, let's append it as a "System" message in local state so next turn sees it.
            
            conv["messages"].append({
                "agent": "system",
                "role": "system",
                "content": tool_output_str,
                "turn": turn_num, # Same turn?
                "timestamp": datetime.now().isoformat()
            })
            
            # We skip DB save for tool outputs for now unless we update the schema, 
            # OR we append it to the agent's message content?
            # Appending to content is safer for current DB schema.
            # Let's append it to the *previous* DB message record?
            # Or just create a new message with role 'agent_a' (self) saying "Tool Result: ..."?
            # That's confusing.
            # Let's just store it in local memory for the context window.
            # The evaluator will see it in the transcript if we pass `conv["messages"]`.

        yield f"data: {json.dumps({'type': 'done', 'turn': turn_num})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/conversation/<conv_id>/analyze', methods=['POST'])
def analyze_conversation(conv_id):
    """Run Kimi K2 analysis AND per-model benchmarking"""
    if conv_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    conv = conversations[conv_id]
    
    # 1. Run Conversation-Level Analysis (Kimi K2)
    analysis_messages = [
        {"role": f"Agent {msg['agent'].upper()}", "content": msg["content"]}
        for msg in conv["messages"]
    ]
    
    analysis = run_analysis(analysis_messages, conv["seed_prompt"])
    conv["analysis"] = analysis
    conv["status"] = "analyzed"
    
    # Update status in DB
    db.update_conversation_status(conv["db_id"], "completed", datetime.now())
    
    # 2. Run Benchmarking (Per-Model Evaluation)
    
    # Construct conversation dict for the evaluator (matches what save_conversation produces)
    conv_data = {
        "id": conv["db_id"],
        "metadata": {
            "seed_prompt": conv["seed_prompt"],
            "end_time": datetime.now().isoformat()
        },
        "agents": {
            "agent_a": {"model": MODELS[conv["model_a"]]["id"]},
            "agent_b": {"model": MODELS[conv["model_b"]]["id"]}
        },
        "messages": [
            {"role": f"agent_{m['agent']}", "content": m['content']} 
            for m in conv["messages"]
        ]
    }

    # Evaluate Agent A
    metrics_a = benchmark_evaluator.evaluate_model_performance(conv_data, "a")
    # Using the friendly name for the DB is better for leaderboard grouping? 
    # Or strict ID? Let's use the key name from backend for grouping (e.g. 'llama-70b') 
    # OR the full model ID. The evaluator returns the ID from 'agents' dict.
    # Let's map it back or just use the ID. The DB schema uses 'model_name'.
    # I'll use the friendly name from MODELS config if available, or the ID.
    model_name_a = MODELS[conv["model_a"]]["name"]
    metrics_a["model_name"] = model_name_a # Override with friendly name for leaderboard
    db.save_model_metrics(metrics_a)
    
    # Evaluate Agent B
    metrics_b = benchmark_evaluator.evaluate_model_performance(conv_data, "b")
    model_name_b = MODELS[conv["model_b"]]["name"]
    metrics_b["model_name"] = model_name_b
    db.save_model_metrics(metrics_b)
    
    # Include metrics in response
    analysis["benchmarks"] = {
        "agent_a": metrics_a,
        "agent_b": metrics_b
    }
    
    return jsonify(analysis)


@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get model leaderboard"""
    stats = db.get_leaderboard()
    return jsonify(stats)


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
