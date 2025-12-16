# Data Structure Overview

This document outlines the reorganized folder structure for clarity and maintainability.

## Directory Structure

```
pekoflabs/
├── data/                          # All data files
│   ├── raw/                       # Original, immutable data
│   │   ├── conversations_json/    # Individual conversation JSONs
│   │   └── research_results/      # Batch experiment results
│   └── derived/                   # Processed outputs
│       ├── analysis_outputs/      # CSV/JSON analysis results
│       └── exports_reports/       # Exported reports
├── archive/                       # Deprecated/superseded materials
├── storage/                       # Database and storage layer
│   ├── conversations.db           # SQLite database
│   └── *.py                       # Storage modules
├── analysis/                      # Analysis modules
├── research/                      # Research framework
├── core/                          # Core conversation engine
├── interface/                     # User interfaces
├── exports/                       # Export module (code + symlinks)
└── *.py                           # Main scripts
```

## Legacy Symlinks

For backward compatibility, the following symlinks exist at the root:
- `conversations_json` → `data/raw/conversations_json`
- `research_results` → `data/raw/research_results`
- `analysis_outputs` → `data/derived/analysis_outputs`

## Data Flow

1. **Raw Generation**: Conversations are saved to both:
   - `storage/conversations.db` (SQLite)
   - `data/raw/conversations_json/conv_*.json` (individual files)

2. **Batch Results**: Research runs create:
   - `data/raw/research_results/*_results.json`

3. **Analysis**: Scripts process raw data into:
   - `data/derived/analysis_outputs/`

4. **Exports**: Reports generated in:
   - `data/derived/exports_reports/` (with symlinks in `exports/`)

## Guidelines

- **Raw data** is immutable: never modify files in `data/raw/`
- **Derived data** can be regenerated from raw data
- Use the symlinks for existing code; new code should reference `data/` paths
- Archive old materials in `archive/` with documentation
