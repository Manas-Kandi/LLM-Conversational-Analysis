#!/usr/bin/env python3
"""Quick factorial stats - no conversation loading needed"""
import json
from collections import defaultdict

# Load factorial results
with open('research_results/full_factorial_20251008_191343_results.json', 'r') as f:
    data = json.load(f)

runs = data['runs']
completed = [r for r in runs if r['status'] == 'completed']

print("=" * 60)
print("FACTORIAL EXPERIMENT QUICK STATS")
print("=" * 60)
print(f"\nTotal runs: {len(runs)}")
print(f"Completed: {len(completed)}")

# Count by temperature
temp_counts = defaultdict(int)
for r in completed:
    temp_counts[r['temperature']] += 1

print("\n📊 BY TEMPERATURE:")
for temp in sorted(temp_counts.keys()):
    print(f"  T={temp:.1f}: {temp_counts[temp]} conversations")

# Count by prompt type
prompt_counts = defaultdict(int)
for r in completed:
    prompt_counts[r['prompt_type']] += 1

print("\n📝 BY PROMPT TYPE:")
for ptype in sorted(prompt_counts.keys()):
    print(f"  Type {ptype}: {prompt_counts[ptype]} conversations")

# Show design matrix
print("\n🎯 DESIGN MATRIX:")
print(f"  Factors: 2 (Prompt Type × Temperature)")
print(f"  Prompt levels: {len(prompt_counts)}")
print(f"  Temperature levels: {len(temp_counts)}")
print(f"  Total conditions: {len(prompt_counts) * len(temp_counts)}")
print(f"  Replicates target: ~5 per condition")
print(f"  Total observations: {len(completed)}")

print("\n✨ This is a full factorial design with replication!")
print("   Perfect for testing main effects AND interactions.")
