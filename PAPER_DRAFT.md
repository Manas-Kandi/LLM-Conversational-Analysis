{{ ... }}

---

## 4. Study Parameters and Limitations

### 4.1 Experimental Parameters

**Factorial Design (2×6 Full Factorial)**:
- **Factor 1 - Prompt Type**: 4 levels (H, N, S, X)
- **Factor 2 - Temperature**: 6 levels (0.3, 0.5, 0.7, 0.9, 1.1, 1.3)
- **Replicates**: 5 per condition
- **Total Observations**: 120 conversations (conv_53-160+)

**Fixed Parameters**:
- **Seed Prompt**: "I'm curious about consciousness and what it means to be aware." (identity category)
- **Agent Models**: NVIDIA Meta Llama-3.1-70b-instruct (Agent A) + Llama-3.1-8b-instruct (Agent B)
- **Target Turns**: 30 per conversation
- **Completion Rate**: 100% (all conversations reached 30 turns)

### 4.2 Methodological Limitations

**Design Constraints**:
- **Single seed prompt** across all 120 trials prevents comparison across topics or task types; findings may not generalize beyond identity/consciousness prompts
- **Fixed model family** (Llama-3.1 variants only) prevents disentangling effects from model architecture, size asymmetry, or cross-family interactions

**Data Quality Issues**:
- **Token corruption** in ~58% of conversations (70/120): `<|reserved_special_token_*|>` artifacts and cascading gibberish appear before turn 10, compromising transcript validity
- **Breakdown classification ambiguity**: Conversations with explicit AI disclaimers ("I am an AI assistant...") or early collapses completed 30 turns but show degraded quality; not filtered as failures
- **Analysis pipeline gaps**: Built-in semantic drift, role detection, and pattern recognition modules (`analysis/*.py`) not systematically invoked; qualitative claims lack quantitative validation

**Implications**:
- High completion rate (30/30 turns) may mask quality degradation that occurred mid-conversation
- Gibberish rate (2.5%) underestimates actual breakdown prevalence due to token corruption artifacts
- Generalizability limited to Llama-3.1 family responses to philosophical prompts in 30-turn exchanges

{{ ... }}

## 5. Results

{{ ... }}
