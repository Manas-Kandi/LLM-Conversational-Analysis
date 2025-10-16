#!/usr/bin/env python3
"""Test NVIDIA NIM API connection"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

nvidia_key = os.getenv("NVIDIA_API_KEY")
nvidia_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

print(f"🔑 NVIDIA API Key: {nvidia_key[:20]}..." if nvidia_key else "❌ No NVIDIA API key")
print(f"🌐 Base URL: {nvidia_url}")

# Test with Llama model
model = "meta/llama-3.1-70b-instruct"
print(f"\n🤖 Testing model: {model}")

try:
    client = OpenAI(
        api_key=nvidia_key,
        base_url=nvidia_url
    )
    
    print("📡 Sending test request...")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        temperature=0.7,
        max_tokens=10
    )
    
    print(f"✅ Success! Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"\nFull error details:")
    import traceback
    traceback.print_exc()
