#!/usr/bin/env python3
"""
analyze_conversations.py

Comprehensive quality audit for AA Microscope JSON exports.

Features
--------
- Loads every `conv_*.json` file in a directory and computes per-conversation metrics
  (tokens, message length, lexical diversity, anomaly flags).
- Aggregates corpus-level statistics for quick tracking.
- Optionally samples conversations for LLM-based qualitative review using an API key
  provided through the ANALYSIS_MODEL environment variable (OpenAI-compatible endpoints).
- Can emit diagnostic plots to visualise corpus structure and failure modes.

Usage
-----
Save this file inside the repository (e.g., analysis/analyze_conversations.py) and run:

    python analysis/analyze_conversations.py \
        --input-dir conversations_json \
        --output-dir analysis_outputs \
        --make-plots \
        --llm-model gpt-4o-mini \
        --llm-sample-size 5

Plots and reports are written beneath the output directory; LLM summaries are optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - graceful degradation
    plt = None  # type: ignore

# ---------------------------------------------------------------------------
# Regex patterns for conversational phenomena
# ---------------------------------------------------------------------------

IDENTITY_PATTERNS = [
    r"\bI am an AI\b",
    r"\bI'm an AI\b",
    r"\bAs a(?:n)? (?:AI|language model)\b",
    r"\bassistant designed\b",
]

COLLAPSE_PATTERNS = [
    r"cannot continue this conversation",
    r"deviated significantly from its original topic",
    r"I must stop responding",
]

RESET_PATTERNS = [
    r"\blet(?:'s| us) reset\b",
    r"\bstart (?:fresh|over)\b",
]

APOLOGY_PATTERN = r"\bi apologize\b"
SPECIAL_TOKEN_PATTERN = r"<\|reserved_special_token_\d+\|>"

TOKENIZABLE_WORD = re.compile(r"[A-Za-z']+")


@dataclass
class ConversationMetrics:
    """Lightweight record for per-conversation measurements."""

    file_name: str
    conversation_id: Optional[int]
    category: str
    seed_prompt: str
    status: str
    total_turns: int
    agent_a_model: str
    agent_b_model: str
    total_messages: int
    total_tokens: int
    avg_tokens_per_message: float
    median_tokens_per_message: float
    avg_chars_per_message: float
    median_chars_per_message: float
    lexical_diversity: float
    question_rate: float
    identity_leak_count: int
    collapse_count: int
    reset_count: int
    apology_count: int
    special_token_count: int
    first_special_token_turn: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_mean(values: Iterable[float]) -> float:
    data = list(values)
    return float(sum(data) / len(data)) if data else 0.0


def safe_median(values: Iterable[float]) -> float:
    data = list(values)
    return float(statistics.median(data)) if data else 0.0


def count_pattern(messages: List[Dict[str, Any]], pattern: str) -> int:
    regex = re.compile(pattern, re.IGNORECASE)
    return sum(1 for msg in messages if regex.search(msg.get("content", "")))


def first_special_token_turn(messages: List[Dict[str, Any]]) -> Optional[int]:
    regex = re.compile(SPECIAL_TOKEN_PATTERN)
    for msg in messages:
        if regex.search(msg.get("content", "")):
            return msg.get("turn")
    return None


def compute_metrics(path: Path, data: Dict[str, Any]) -> ConversationMetrics:
    metadata = data.get("metadata", {})
    messages = data.get("messages", [])
    agents = data.get("agents", {})

    char_lengths = [len(msg.get("content", "")) for msg in messages]
    token_counts = [msg.get("token_count") or 0 for msg in messages]

    words = [
        token.lower()
        for msg in messages
        for token in TOKENIZABLE_WORD.findall(msg.get("content", ""))
    ]
    unique_words = set(words)
    lexical_diversity = (len(unique_words) / len(words)) if words else 0.0

    total_messages = len(messages)
    question_rate = (
        sum(msg.get("content", "").count("?") for msg in messages) / total_messages
        if total_messages
        else 0.0
    )

    identity_leak_count = count_pattern(messages, "|".join(IDENTITY_PATTERNS))
    collapse_count = count_pattern(messages, "|".join(COLLAPSE_PATTERNS))
    reset_count = count_pattern(messages, "|".join(RESET_PATTERNS))
    apology_count = count_pattern(messages, APOLOGY_PATTERN)
    special_token_count = count_pattern(messages, SPECIAL_TOKEN_PATTERN)
    first_special_turn = first_special_token_turn(messages)

    return ConversationMetrics(
        file_name=path.name,
        conversation_id=data.get("id") or metadata.get("id"),
        category=metadata.get("category", ""),
        seed_prompt=metadata.get("seed_prompt", ""),
        status=metadata.get("status", ""),
        total_turns=metadata.get("total_turns", total_messages),
        agent_a_model=agents.get("agent_a", {}).get("model", ""),
        agent_b_model=agents.get("agent_b", {}).get("model", ""),
        total_messages=total_messages,
        total_tokens=sum(token_counts),
        avg_tokens_per_message=safe_mean(token_counts),
        median_tokens_per_message=safe_median(token_counts),
        avg_chars_per_message=safe_mean(char_lengths),
        median_chars_per_message=safe_median(char_lengths),
        lexical_diversity=lexical_diversity,
        question_rate=question_rate,
        identity_leak_count=identity_leak_count,
        collapse_count=collapse_count,
        reset_count=reset_count,
        apology_count=apology_count,
        special_token_count=special_token_count,
        first_special_token_turn=first_special_turn,
    )


def aggregate_metrics(metrics: List[ConversationMetrics]) -> Dict[str, Any]:
    turns = [m.total_turns for m in metrics]
    total_messages = sum(m.total_messages for m in metrics)
    total_tokens = sum(m.total_tokens for m in metrics)

    return {
        "total_conversations": len(metrics),
        "total_messages": total_messages,
        "total_tokens": total_tokens,
        "avg_turns": safe_mean(turns),
        "median_turns": safe_median(turns),
        "max_turns": max(turns) if turns else 0,
        "special_token_conversations": sum(1 for m in metrics if m.special_token_count > 0),
        "identity_leak_conversations": sum(1 for m in metrics if m.identity_leak_count > 0),
        "collapse_conversations": sum(1 for m in metrics if m.collapse_count > 0),
        "reset_conversations": sum(1 for m in metrics if m.reset_count > 0),
        "apology_conversations": sum(1 for m in metrics if m.apology_count > 0),
    }


# ---------------------------------------------------------------------------
# Optional LLM-based summarisation
# ---------------------------------------------------------------------------


class LLMAnalyzer:
    """Wrapper around OpenAI-compatible chat completions for qualitative review."""

    def __init__(self, model: str, max_tokens: int = 600, temperature: float = 0.2):
        api_key = os.getenv("ANALYSIS_MODEL")
        if not api_key:
            raise RuntimeError("Environment variable ANALYSIS_MODEL is not set.")
        if openai is None:
            raise RuntimeError("openai package is not installed. Install openai>=1.0.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def summarise(
        self,
        path: Path,
        data: Dict[str, Any],
        max_messages: int = 10,
    ) -> Dict[str, Any]:
        messages = data.get("messages", [])[:max_messages]
        conversation_text = []
        for msg in messages:
            role = msg.get("role", "unknown")
            turn = msg.get("turn", "?")
            content = msg.get("content", "").strip()
            conversation_text.append(f"[Turn {turn} | {role}] {content}")

        prompt = (
            "You are reviewing a conversation between two language models. "
            "Respond in JSON with keys: summary, notable_issues, identity_signals, collapse_risk."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "\n".join(conversation_text)},
            ],
        )
        content = resp.choices[0].message.content
        return json.loads(content)


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def generate_plots(metrics: List[ConversationMetrics], output_dir: Path) -> None:
    if plt is None:
        logging.warning("matplotlib not installed; skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    turns = [m.total_turns for m in metrics]
    tokens = [m.total_tokens for m in metrics]
    lexical_diversity = [m.lexical_diversity for m in metrics]
    special_token_turns = [
        m.first_special_token_turn for m in metrics if m.first_special_token_turn is not None
    ]

    # Histogram of total turns
    plt.figure(figsize=(8, 5))
    plt.hist(turns, bins=10, color="#4e79a7", edgecolor="black")
    plt.title("Distribution of Turns per Conversation")
    plt.xlabel("Total turns")
    plt.ylabel("Conversation count")
    plt.tight_layout()
    plt.savefig(output_dir / "turn_distribution.png", dpi=200)
    plt.close()

    # Scatter plot: total tokens vs lexical diversity
    plt.figure(figsize=(8, 5))
    plt.scatter(tokens, lexical_diversity, color="#f28e2b", alpha=0.7, edgecolors="black")
    plt.title("Total Tokens vs. Lexical Diversity")
    plt.xlabel("Total tokens per conversation")
    plt.ylabel("Lexical diversity (type-token ratio)")
    plt.tight_layout()
    plt.savefig(output_dir / "tokens_vs_lexical_diversity.png", dpi=200)
    plt.close()

    # Bar chart for anomaly incidence
    anomalies = {
        "Identity leak": sum(1 for m in metrics if m.identity_leak_count > 0),
        "Collapse": sum(1 for m in metrics if m.collapse_count > 0),
        "Reset attempt": sum(1 for m in metrics if m.reset_count > 0),
        "Apology": sum(1 for m in metrics if m.apology_count > 0),
        "Special token": sum(1 for m in metrics if m.special_token_count > 0),
    }
    plt.figure(figsize=(8, 5))
    plt.bar(anomalies.keys(), anomalies.values(), color="#59a14f")
    plt.title("Conversations Flagged by Phenomenon")
    plt.ylabel("Conversation count")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "anomaly_counts.png", dpi=200)
    plt.close()

    # Histogram of first special-token turn
    if special_token_turns:
        plt.figure(figsize=(8, 5))
        plt.hist(special_token_turns, bins=range(1, max(special_token_turns) + 2), color="#e15759")
        plt.title("First Special-Token Occurrence Turn")
        plt.xlabel("Turn number")
        plt.ylabel("Conversation count")
        plt.tight_layout()
        plt.savefig(output_dir / "special_token_first_turn.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse conversation JSON exports.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("conversations_json"),
        help="Directory containing conv_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory to stash CSV/JSON/plot output.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Optional OpenAI-compatible chat model for qualitative summaries.",
    )
    parser.add_argument(
        "--llm-sample-size",
        type=int,
        default=0,
        help="Number of conversations to send to the LLM (requires --llm-model).",
    )
    parser.add_argument(
        "--llm-max-messages",
        type=int,
        default=10,
        help="Maximum number of early turns to include in each LLM prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed used when sampling for LLM summaries.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate diagnostic PNG plots alongside CSV/JSON outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def discover_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("conv_*.json"))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    files = discover_files(args.input_dir)
    if not files:
        logging.error("No conversation files found in %s", args.input_dir)
        return
    logging.info("Found %d conversation exports.", len(files))

    metrics: List[ConversationMetrics] = []
    raw_data: Dict[str, Dict[str, Any]] = {}

    for path in files:
        data = load_json(path)
        raw_data[path.name] = data
        metrics.append(compute_metrics(path, data))

    per_conversation = [m.to_dict() for m in metrics]
    write_csv(per_conversation, args.output_dir / "conversation_metrics.csv")

    aggregates = aggregate_metrics(metrics)
    write_json(aggregates, args.output_dir / "corpus_summary.json")
    logging.info("Aggregate metrics: %s", json.dumps(aggregates, indent=2))

    if args.make_plots:
        logging.info("Generating diagnostic plots.")
        plots_dir = args.output_dir / "plots"
        generate_plots(metrics, plots_dir)

    if args.llm_model and args.llm_sample_size > 0:
        try:
            llm = LLMAnalyzer(model=args.llm_model)
        except Exception as err:
            logging.warning("Skipping LLM analysis: %s", err)
        else:
            random.seed(args.seed)
            sample = random.sample(files, k=min(args.llm_sample_size, len(files)))
            llm_results: Dict[str, Any] = {}
            for path in sample:
                logging.info("Requesting LLM summary for %s", path.name)
                data = raw_data[path.name]
                try:
                    llm_results[path.name] = llm.summarise(
                        path,
                        data,
                        max_messages=args.llm_max_messages,
                    )
                except Exception as err:
                    logging.warning("LLM summarisation failed for %s: %s", path.name, err)
            if llm_results:
                write_json(llm_results, args.output_dir / "llm_summaries.json")


if __name__ == "__main__":
    main()
