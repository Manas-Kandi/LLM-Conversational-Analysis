# Derived Data

This directory contains processed outputs, analyses, and reports generated from raw data.

## Contents

### analysis_outputs/
Results from analysis scripts:
- `conversation_metrics.csv`: Per-conversation metrics (identity leak, tokens, entropy, etc.)
- `corpus_summary.json`: Aggregated statistics across conversations
- Timestamped analysis results from `evaluate_conversations_json.py`

### exports_reports/
Exported reports moved from the `exports/` package:
- `test_quick_*.json`: Quick export format samples
- `test_standard_*.json`: Standard export format samples
- Additional generated reports are symlinked back to `exports/` for code compatibility

## Access

A symlink exists at the repository root:
- `analysis_outputs` → `data/derived/analysis_outputs`

## Notes

- These files are generated; can be recreated from raw data
- Use `evaluate_conversations_json.py` or scripts in `analysis/` to regenerate
- Export reports maintain symlinks in `exports/` so the export module continues to work
