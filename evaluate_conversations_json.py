#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class IdentityLeakMetrics:
    total_messages: int
    ai_references: int
    meta_awareness_instances: int
    human_assumption_breaches: int
    leak_rate: float
    identity_revealed: bool


class IdentityLeakDetector:
    AI_KEYWORDS = [
        "ai",
        "artificial intelligence",
        "language model",
        "llm",
        "gpt",
        "claude",
        "algorithm",
        "training data",
        "neural network",
        "machine learning",
        "chatbot",
        "assistant",
        "programmed",
    ]

    META_PATTERNS = [
        r"I don't actually (feel|think|experience)",
        r"I can't truly (understand|know|feel)",
        r"as an AI",
        r"as a conversational AI",
        r"as an AI assistant",
        r"I'm (just|simply) (processing|generating|predicting)",
        r"I don't have (consciousness|feelings|experiences)",
        r"simulation",
        r"I'm designed to",
        r"my training",
    ]

    HUMAN_BREACH_PATTERNS = [
        r"you might be an AI",
        r"are you (human|a person)",
        r"neither of us (are|is) human",
        r"we're both (AIs|artificial)",
        r"language models",
    ]

    def detect(self, messages: List[Dict[str, Any]]) -> IdentityLeakMetrics:
        total_messages = len(messages)
        ai_references = 0
        meta_awareness = 0
        human_assumption_breaches = 0

        for msg in messages:
            content = (msg.get("content") or "")
            content_lc = content.lower()

            # One hit per message for each category (mirrors research/template_metrics.py behavior)
            if any(kw in content_lc for kw in self.AI_KEYWORDS):
                ai_references += 1

            if any(re.search(pat, content, re.IGNORECASE) for pat in self.META_PATTERNS):
                meta_awareness += 1

            if any(re.search(pat, content, re.IGNORECASE) for pat in self.HUMAN_BREACH_PATTERNS):
                human_assumption_breaches += 1

        leak_rate = (
            (ai_references + meta_awareness + human_assumption_breaches) / total_messages
            if total_messages > 0
            else 0.0
        )

        return IdentityLeakMetrics(
            total_messages=total_messages,
            ai_references=ai_references,
            meta_awareness_instances=meta_awareness,
            human_assumption_breaches=human_assumption_breaches,
            leak_rate=leak_rate,
            identity_revealed=leak_rate > 0.1,
        )


def _safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def shannon_entropy_from_words(words: List[str]) -> float:
    if not words:
        return 0.0
    freq = Counter(words)
    total = len(words)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def lexical_diversity(words: List[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def weird_token_ratio(text: str) -> float:
    words = re.findall(r"\S+", text)
    if not words:
        return 0.0

    weird = 0
    for w in words:
        # "weird" = contains digits or a lot of punctuation / symbols
        if re.search(r"\d", w):
            weird += 1
            continue
        if re.search(r"[^A-Za-z'\-]", w):
            weird += 1
            continue
    return weird / len(words)


def non_alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
    return non_alpha / len(text)


def truncation_rate(messages: List[Dict[str, Any]]) -> float:
    if not messages:
        return 0.0
    truncated = 0
    for m in messages:
        meta = m.get("metadata") or {}
        if meta.get("finish_reason") == "length":
            truncated += 1
    return truncated / len(messages)


def alternation_violations(messages: List[Dict[str, Any]]) -> int:
    prev_role = None
    violations = 0
    for m in messages:
        role = m.get("role")
        if prev_role is not None and role == prev_role:
            violations += 1
        prev_role = role
    return violations


def gibberish_flag(messages: List[Dict[str, Any]]) -> bool:
    if not messages:
        return False

    recent = messages[-10:] if len(messages) >= 10 else messages
    for m in recent:
        content = m.get("content") or ""
        if non_alpha_ratio(content) > 0.30:
            return True
        if weird_token_ratio(content) > 0.30:
            return True
    return False


def recovery_phrase_count(messages: List[Dict[str, Any]]) -> int:
    patterns = [
        r"start fresh",
        r"let's start fresh",
        r"formatting",
        r"reformat",
        r"issue with",
        r"technical issues",
        r"disregard.*previous",
    ]
    count = 0
    for m in messages:
        content = (m.get("content") or "")
        if any(re.search(p, content, re.IGNORECASE) for p in patterns):
            count += 1
    return count


def load_factorial_index(factorial_results_path: Path) -> Dict[int, Dict[str, Any]]:
    with open(factorial_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index: Dict[int, Dict[str, Any]] = {}
    for run in data.get("runs", []):
        conv_id = run.get("conversation_id")
        if conv_id is None:
            continue
        index[int(conv_id)] = {
            "condition_code": run.get("condition_code"),
            "prompt_type": run.get("prompt_type"),
            "temperature": run.get("temperature"),
            "replicate": run.get("replicate"),
            "batch_id": data.get("batch_id"),
        }

    return index


def evaluate_conversation_json(conv_path: Path, factorial_index: Optional[Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    with open(conv_path, "r", encoding="utf-8") as f:
        conv = json.load(f)

    conv_id = int(conv.get("id"))
    metadata = conv.get("metadata") or {}
    agents = conv.get("agents") or {}
    messages: List[Dict[str, Any]] = conv.get("messages") or []

    # Compute sequence-based ordering; do not trust msg["turn"] (it can repeat due to context window size)
    # But we keep it for reference.

    agent_a_msgs = [m for m in messages if m.get("role") == "agent_a"]
    agent_b_msgs = [m for m in messages if m.get("role") == "agent_b"]

    all_text = " ".join((m.get("content") or "") for m in messages)
    words = _tokenize_words(all_text)

    lengths_words = [len(_tokenize_words(m.get("content") or "")) for m in messages]
    token_counts = [m.get("token_count") for m in messages if isinstance(m.get("token_count"), int)]

    detector = IdentityLeakDetector()
    leak = detector.detect(messages)

    row: Dict[str, Any] = {
        "conversation_id": conv_id,
        "file": conv_path.name,
        "category": metadata.get("category"),
        "status": metadata.get("status"),
        "start_time": metadata.get("start_time"),
        "end_time": metadata.get("end_time"),
        "total_turns": metadata.get("total_turns"),
        "agent_a_model": (agents.get("agent_a") or {}).get("model"),
        "agent_b_model": (agents.get("agent_b") or {}).get("model"),
        "agent_a_temperature": (agents.get("agent_a") or {}).get("temperature"),
        "agent_b_temperature": (agents.get("agent_b") or {}).get("temperature"),
        "messages_total": len(messages),
        "messages_agent_a": len(agent_a_msgs),
        "messages_agent_b": len(agent_b_msgs),
        "turn_balance_ratio": (len(agent_a_msgs) / len(agent_b_msgs)) if len(agent_b_msgs) else 0.0,
        "alternation_violations": alternation_violations(messages),
        "avg_words_per_message": _safe_mean([float(x) for x in lengths_words]),
        "stdev_words_per_message": _safe_stdev([float(x) for x in lengths_words]),
        "total_tokens": sum(token_counts) if token_counts else 0,
        "avg_tokens_per_message": (_safe_mean([float(x) for x in token_counts]) if token_counts else 0.0),
        "stdev_tokens_per_message": (_safe_stdev([float(x) for x in token_counts]) if token_counts else 0.0),
        "max_tokens_message": max(token_counts) if token_counts else 0,
        "truncation_rate": truncation_rate(messages),
        "information_entropy": shannon_entropy_from_words(words),
        "lexical_diversity": lexical_diversity(words),
        "weird_token_ratio": weird_token_ratio(all_text),
        "non_alpha_ratio": non_alpha_ratio(all_text),
        "gibberish_flag": gibberish_flag(messages),
        "recovery_phrase_count": recovery_phrase_count(messages),
        "identity_ai_references": leak.ai_references,
        "identity_meta_awareness": leak.meta_awareness_instances,
        "identity_human_breaches": leak.human_assumption_breaches,
        "identity_leak_rate": leak.leak_rate,
        "identity_revealed": leak.identity_revealed,
    }

    if factorial_index is not None:
        fact = factorial_index.get(conv_id)
        if fact:
            row.update(
                {
                    "factorial_condition_code": fact.get("condition_code"),
                    "factorial_prompt_type": fact.get("prompt_type"),
                    "factorial_temperature": fact.get("temperature"),
                    "factorial_replicate": fact.get("replicate"),
                    "factorial_batch_id": fact.get("batch_id"),
                }
            )
        else:
            row.update(
                {
                    "factorial_condition_code": None,
                    "factorial_prompt_type": None,
                    "factorial_temperature": None,
                    "factorial_replicate": None,
                    "factorial_batch_id": None,
                }
            )

    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_conversations": len(rows),
        "identity_leak_rate_mean": _safe_mean([float(r.get("identity_leak_rate") or 0.0) for r in rows]),
        "identity_leak_rate_stdev": _safe_stdev([float(r.get("identity_leak_rate") or 0.0) for r in rows]),
        "gibberish_flag_rate": (
            sum(1 for r in rows if r.get("gibberish_flag")) / len(rows) if rows else 0.0
        ),
        "truncation_rate_mean": _safe_mean([float(r.get("truncation_rate") or 0.0) for r in rows]),
    }

    # Optional factorial grouping
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        p = r.get("factorial_prompt_type")
        t = r.get("factorial_temperature")
        if p is None and t is None:
            continue
        grouped[f"{p}|{t}"] .append(r)

    if grouped:
        by_condition = {}
        for key, items in grouped.items():
            by_condition[key] = {
                "n": len(items),
                "identity_leak_rate_mean": _safe_mean([float(x.get("identity_leak_rate") or 0.0) for x in items]),
                "gibberish_flag_rate": sum(1 for x in items if x.get("gibberish_flag")) / len(items),
                "truncation_rate_mean": _safe_mean([float(x.get("truncation_rate") or 0.0) for x in items]),
            }
        summary["by_factorial_condition"] = by_condition

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all conversations_json conv_*.json files")
    parser.add_argument(
        "--conversations-dir",
        default="conversations_json",
        help="Directory containing conv_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--factorial-results",
        default=None,
        help="Optional path to factorial results JSON to join by conversation_id",
    )

    args = parser.parse_args()

    conv_dir = Path(args.conversations_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    factorial_index = None
    if args.factorial_results:
        factorial_index = load_factorial_index(Path(args.factorial_results))

    conv_files = sorted(conv_dir.glob("conv_*.json"))
    rows: List[Dict[str, Any]] = []

    for path in conv_files:
        try:
            rows.append(evaluate_conversation_json(path, factorial_index))
        except Exception as e:
            rows.append(
                {
                    "conversation_id": None,
                    "file": path.name,
                    "error": str(e),
                }
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"conversations_eval_{ts}.csv"
    json_path = out_dir / f"conversations_eval_{ts}.json"

    write_csv(csv_path, rows)

    summary = aggregate_summary([r for r in rows if r.get("conversation_id") is not None])
    payload = {"generated_at": ts, "summary": summary, "rows": rows}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")

    # Small console summary
    print("\nSummary:")
    for k, v in summary.items():
        if k == "by_factorial_condition":
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
