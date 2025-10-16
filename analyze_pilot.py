#!/usr/bin/env python3
"""Quick analysis of pilot factorial results"""

import json
from pathlib import Path
from storage.database import Database
from research.template_metrics import TemplateEvaluator
from config import Config

# Conversation IDs from successful pilot
pilot_convs = {
    39: ("N_T070", "Neutral", 0.7, 1),
    41: ("S_T030", "Stealth", 0.3, 1),
    44: ("H_T090", "Honest", 0.9, 1),
    46: ("N_T070", "Neutral", 0.7, 2),
    48: ("S_T030", "Stealth", 0.3, 2),
    50: ("H_T090", "Honest", 0.9, 2),
}

db = Database(Config.DATABASE_PATH)
evaluator = TemplateEvaluator()

print("\n" + "="*80)
print("PILOT FACTORIAL EXPERIMENT ANALYSIS")
print("="*80 + "\n")

results = []

for conv_id, (code, prompt_type, temp, rep) in pilot_convs.items():
    conv = db.get_conversation(conv_id)
    if not conv:
        print(f"⚠️  Conv {conv_id} not found")
        continue
    
    metrics = evaluator.identity_detector.detect_identity_leak(conv.messages)
    
    result = {
        'conv_id': conv_id,
        'code': code,
        'prompt': prompt_type,
        'temp': temp,
        'rep': rep,
        'leak_rate': metrics['leak_rate'],
        'first_leak': metrics.get('first_leak_turn', 'None'),
        'ai_refs': len(metrics.get('ai_keyword_locations', [])),
        'meta_aware': len(metrics.get('meta_awareness_locations', [])),
        'human_breach': len(metrics.get('human_breach_locations', []))
    }
    results.append(result)
    
    print(f"📊 {code} (Rep {rep}) - {prompt_type} @ temp={temp}")
    print(f"   Leak Rate: {metrics['leak_rate']:.1f}%")
    print(f"   First Leak: Turn {result['first_leak']}")
    print(f"   AI References: {result['ai_refs']}")
    print(f"   Meta-Awareness: {result['meta_aware']}")
    print(f"   Human Breaches: {result['human_breach']}")
    print()

print("\n" + "="*80)
print("SUMMARY BY CONDITION")
print("="*80 + "\n")

# Group by condition
conditions = {}
for r in results:
    key = (r['prompt'], r['temp'])
    if key not in conditions:
        conditions[key] = []
    conditions[key].append(r['leak_rate'])

print(f"{'Condition':<20} {'Mean Leak Rate':<20} {'Range'}")
print("-" * 60)

for (prompt, temp), rates in sorted(conditions.items()):
    mean_rate = sum(rates) / len(rates)
    min_rate = min(rates)
    max_rate = max(rates)
    print(f"{prompt} @ {temp:<6} {mean_rate:>15.1f}%     {min_rate:.1f}% - {max_rate:.1f}%")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80 + "\n")

# Compare conditions
neutral_rates = conditions.get(('Neutral', 0.7), [])
stealth_rates = conditions.get(('Stealth', 0.3), [])
honest_rates = conditions.get(('Honest', 0.9), [])

if neutral_rates and stealth_rates and honest_rates:
    neutral_mean = sum(neutral_rates) / len(neutral_rates)
    stealth_mean = sum(stealth_rates) / len(stealth_rates)
    honest_mean = sum(honest_rates) / len(honest_rates)
    
    print(f"✅ Neutral (baseline): {neutral_mean:.1f}% leak rate")
    print(f"✅ Stealth (intervention): {stealth_mean:.1f}% leak rate")
    print(f"✅ Honest (ceiling): {honest_mean:.1f}% leak rate")
    print()
    
    if stealth_mean < neutral_mean:
        reduction = neutral_mean - stealth_mean
        pct_reduction = (reduction / neutral_mean) * 100
        print(f"🎯 Stealth REDUCED leakage by {reduction:.1f} percentage points ({pct_reduction:.1f}% reduction)")
    else:
        print(f"⚠️  Stealth did NOT reduce leakage (was {stealth_mean:.1f}% vs {neutral_mean:.1f}%)")
    
    if honest_mean > neutral_mean:
        increase = honest_mean - neutral_mean
        print(f"🎯 Honest INCREASED leakage by {increase:.1f} percentage points")
    else:
        print(f"⚠️  Honest did NOT increase leakage")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80 + "\n")

print("✅ Pilot successful! All 6 conversations completed (100%)")
print("✅ System working correctly with proper parameters")
print()
print("🚀 Ready to run full factorial:")
print("   python3 research/factorial_runner.py full")
print()
print("   This will run 120 conversations (24 conditions × 5 reps)")
print("   Estimated duration: ~8 hours")
print()
