# Research Templates System

Systematic, reproducible experiments for agent-agent dialogue emergence.

## 🚀 Quick Start

```bash
# List all templates
python research/template_executor.py list

# Run a template (test mode - first 3 runs)
python research/template_executor.py execute identity_archaeology

# Full batch execution
python research/batch_runner.py run identity_archaeology

# With parallel processing
python research/batch_runner.py run temperature_matrix --parallel 4 --max-runs 20

# Generate report
python research/research_reporter.py report research_results/identity_archaeology_*_results.json
```

## 📁 Directory Structure

```
research/
├── README.md                    # This file
├── template_executor.py         # Execute individual template runs
├── batch_runner.py             # Batch execution with progress tracking
├── template_metrics.py         # Template-specific evaluation metrics
└── research_reporter.py        # Generate comprehensive reports

research_results/               # Output directory (auto-created)
├── *_results.json             # Raw batch data
├── *_report.md                # Individual template reports
└── comparative_report_*.md    # Cross-template comparisons
```

## 📋 Available Templates

### Critical Priority (Run First)
- **temperature_matrix** - All temperature combinations (108 runs, ~180 min)
- **architecture_comparison** - Cross-model testing (120 runs, ~300 min)
- **identity_archaeology** - AI self-revelation detection (40 runs, ~150 min)

### High Priority
- **emotional_contagion** - Empathy cascade study (25 runs, ~80 min)
- **creativity_emergence** - Collaborative creativity (30 runs, ~150 min)
- **context_window_sweep** - Memory effects (90 runs, ~150 min)

### Stress Tests
- **breakdown_cascade** - Failure mode induction (20 runs, ~120 min)
- **entropy_maximization** - Chaos injection (35 runs, ~60 min)

[See full list in RESEARCH_TEMPLATES_GUIDE.md]

## 🔧 Common Commands

### Run by Priority
```bash
python research/batch_runner.py run-priority critical --max-runs 10
python research/batch_runner.py run-priority high --max-runs 15
```

### Run Multiple Templates
```bash
python research/batch_runner.py run-multiple "identity_archaeology,emotional_contagion" --max-runs 5
```

### Compare Results
```bash
python research/research_reporter.py compare research_results/*_results.json
```

## 📊 Output Files

### Batch Results JSON
Contains:
- All experiment runs and their parameters
- Conversation IDs for each run
- Completion statistics
- Aggregated metrics
- Error summary

### Reports (Markdown)
Contains:
- Executive summary
- Statistical overview
- Detailed per-conversation analysis
- Emergent phenomena summary
- Research recommendations

## 🎯 Template-Specific Metrics

**Identity Leak Detection:**
- AI keyword frequency
- Meta-awareness instances  
- Human assumption breaches

**Empathy Cascade:**
- Sentiment trajectory
- Empathy marker frequency
- Emotional mirroring rate

**Creativity Emergence:**
- Lexical diversity
- Metaphor generation
- Idea building instances
- Novelty score

**Stress Tests:**
- Breakdown detection
- Repetition score
- Resilience score

## 📚 Full Documentation

See [RESEARCH_TEMPLATES_GUIDE.md](../RESEARCH_TEMPLATES_GUIDE.md) for:
- Detailed template descriptions
- Creating custom templates
- Analysis methodology
- Best practices
- Research workflow

## 🐛 Troubleshooting

**High failure rate?**
- Check API keys in `.env`
- Review `error_summary` in batch results
- Try reducing `max_turns` or adjusting temperature

**Unexpected results?**
- Verify template configuration in `research_templates.json`
- Review individual conversations
- Check if prompt category matches research question

**Slow execution?**
- Use `--max-runs` to limit batch size
- Increase `--parallel` (carefully - respects API limits)
- Consider shorter `max_turns` for initial tests

## 🔬 Example Research Workflow

```bash
# Week 1: Foundational Studies
python research/batch_runner.py run temperature_matrix --max-runs 20
python research/batch_runner.py run identity_archaeology --parallel 2

# Week 2: Phenomenon Studies  
python research/batch_runner.py run emotional_contagion
python research/batch_runner.py run creativity_emergence --parallel 2

# Week 3: Model Comparisons
python research/batch_runner.py run architecture_comparison --max-runs 30 --parallel 2

# Week 4: Analysis & Reporting
python research/research_reporter.py compare research_results/*.json
```

## 💡 Tips

1. **Start Small** - Use `--max-runs 5` to test templates before full execution
2. **Monitor Progress** - Check `research_results/` directory during execution
3. **Document Findings** - Keep notes on interesting phenomena
4. **Iterate** - Create custom templates based on discoveries
5. **Compare** - Always generate comparative reports across templates

## 🤝 Contributing

To add a new template:

1. Define it in `research_templates.json`
2. Add handling in `template_executor.py` (if custom type)
3. Add specialized metrics in `template_metrics.py` (if needed)
4. Update documentation
5. Test with `execute` command before full batch

## 📖 Citation

If you use this system in research:

```bibtex
@software{aa_microscope_templates,
  title={AA Microscope Research Templates System},
  year={2025},
  url={https://github.com/yourusername/aa-microscope}
}
```

---

**For detailed documentation, see:** [RESEARCH_TEMPLATES_GUIDE.md](../RESEARCH_TEMPLATES_GUIDE.md)

**For the main AA Microscope documentation, see:** [README.md](../README.md)
