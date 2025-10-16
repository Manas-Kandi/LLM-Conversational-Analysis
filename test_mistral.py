#!/usr/bin/env python3
"""Test Mistral models"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url=os.getenv("NVIDIA_BASE_URL")
)

# Try different Mistral model names
mistral_models = [
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "mixtral-8x7b-instruct-v0.1",
]

for model in mistral_models:
    print(f"\n🧪 Testing: {model}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=5
        )
        print(f"   ✅ SUCCESS: {response.choices[0].message.content}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}")
