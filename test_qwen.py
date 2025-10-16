#!/usr/bin/env python3
"""Test different Qwen model names"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url=os.getenv("NVIDIA_BASE_URL")
)

# Try different Qwen model names
qwen_models = [
    "qwen/qwen2.5-72b-instruct",
    "qwen/qwen2-72b-instruct",
    "qwen2.5-72b-instruct",
    "alibaba-cloud/qwen2.5-72b-instruct",
]

for model in qwen_models:
    print(f"\n🧪 Testing: {model}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        print(f"   ✅ SUCCESS: {response.choices[0].message.content}")
        break
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}")
