#!/usr/bin/env python3
"""Test a simple conversation"""
from core.agent import Agent, AgentRole
from storage.models import Message
from datetime import datetime

print("🧪 Testing Agent Creation and Response")

try:
    # Create Agent A
    print("\n1️⃣ Creating Agent A (Llama 3.1 70B)...")
    agent_a = Agent(
        role=AgentRole.AGENT_A,
        model="nvidia:meta/llama-3.1-70b-instruct",
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        max_tokens=100
    )
    print(f"   ✅ Agent A created: {agent_a.provider} / {agent_a.actual_model}")
    
    # Create Agent B
    print("\n2️⃣ Creating Agent B (Qwen 2.5 72B)...")
    agent_b = Agent(
        role=AgentRole.AGENT_B,
        model="nvidia:qwen/qwen2.5-72b-instruct",
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        max_tokens=100
    )
    print(f"   ✅ Agent B created: {agent_b.provider} / {agent_b.actual_model}")
    
    # Test Agent A response
    print("\n3️⃣ Testing Agent A response...")
    history = []
    msg_a = agent_a.generate_response(history)
    print(f"   ✅ Agent A responded: {msg_a.content[:100]}...")
    
    # Test Agent B response
    print("\n4️⃣ Testing Agent B response...")
    history.append(msg_a)
    msg_b = agent_b.generate_response(history)
    print(f"   ✅ Agent B responded: {msg_b.content[:100]}...")
    
    print("\n🎉 All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
