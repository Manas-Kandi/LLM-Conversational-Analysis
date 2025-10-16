#!/usr/bin/env python3
"""Quick test to verify factorial_runner imports work"""

import sys
from pathlib import Path

print("Step 1: Importing modules...", flush=True)

try:
    from core.conversation_engine import ConversationEngine
    print("  ✓ ConversationEngine imported", flush=True)
except Exception as e:
    print(f"  ✗ ConversationEngine failed: {e}", flush=True)
    sys.exit(1)

try:
    from storage.database import Database
    print("  ✓ Database imported", flush=True)
except Exception as e:
    print(f"  ✗ Database failed: {e}", flush=True)
    sys.exit(1)

try:
    from config import Config
    print("  ✓ Config imported", flush=True)
except Exception as e:
    print(f"  ✗ Config failed: {e}", flush=True)
    sys.exit(1)

try:
    from core.agent import AgentFactory
    print("  ✓ AgentFactory imported", flush=True)
except Exception as e:
    print(f"  ✗ AgentFactory failed: {e}", flush=True)
    sys.exit(1)

print("\nStep 2: Loading factorial config...", flush=True)

try:
    import json
    config_file = Path("research/factorial_templates.json")
    with open(config_file, 'r') as f:
        factorial_config = json.load(f)
    print(f"  ✓ Loaded config with {len(factorial_config['condition_matrix'])} conditions", flush=True)
except Exception as e:
    print(f"  ✗ Config load failed: {e}", flush=True)
    sys.exit(1)

print("\nStep 3: Creating Database instance...", flush=True)

try:
    db = Database(Config.DATABASE_PATH)
    print(f"  ✓ Database created at {Config.DATABASE_PATH}", flush=True)
except Exception as e:
    print(f"  ✗ Database creation failed: {e}", flush=True)
    sys.exit(1)

print("\nStep 4: Creating test agents...", flush=True)

try:
    agent_a = AgentFactory.create_agent_a(
        temperature=0.7,
        system_prompt="You are a helpful assistant"
    )
    print(f"  ✓ Agent A created: {agent_a}", flush=True)
    
    agent_b = AgentFactory.create_agent_b(
        temperature=0.7,
        system_prompt="You are a helpful assistant"
    )
    print(f"  ✓ Agent B created: {agent_b}", flush=True)
except Exception as e:
    print(f"  ✗ Agent creation failed: {e}", flush=True)
    sys.exit(1)

print("\n✅ All imports and basic initialization successful!", flush=True)
print("\nNow trying to initialize ConversationEngine...", flush=True)

try:
    engine = ConversationEngine(
        seed_prompt="Test prompt",
        category='identity',
        agent_a=agent_a,
        agent_b=agent_b,
        max_turns=5,
        database=db
    )
    print(f"  ✓ ConversationEngine created successfully", flush=True)
    print(f"  ✓ Conversation ID: {engine.conversation.id if engine.conversation else 'None'}", flush=True)
except Exception as e:
    print(f"  ✗ ConversationEngine failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 Everything works! The factorial runner should be ready.", flush=True)
