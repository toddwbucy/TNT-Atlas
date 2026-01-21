# Training Analysis: Atlas 58M on FineWeb-Edu 100M

**Run ID**: `atlas_58m_fineweb_100M`
**Started**: 2026-01-20
**Status**: LIKELY FAILURE - Memory system non-functional (continuing to 50K for data)

> **CRITICAL FINDING (2026-01-21)**: Diagnostic analysis revealed the tanh gates in the QK projection are completely dead (operating in linear region). The memory augmentation mechanism is bypassed. See [Failure Analysis](#failure-analysis-2026-01-21) below.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model Size | 58M parameters |
| Architecture | Atlas with TNT (6 layers, d_model=576) |
| Dataset | fineweb-edu (sample-10BT) |
| Tokenizer | SmolLM-135M (vocab=49152) |
| Total Tokens | 100M target |
| Seed Model | SmolLM-135M (ad hoc level stacking) |
| Hardware | GPU0 (training), GPU1 (validation) |

## Training Progress

### Latest Metrics (Step 29,000)

| Metric | Value |
|--------|-------|
| Train Loss | 6.875 |
| Val Loss | 6.568 |
| Val PPL | 712.2 |
| Tokens Processed | 14.8M |

### PPL Evolution

![PPL Chart with Variance](../runs/ppl_chart_smoothed.png)

**Key Observation**: Train PPL > Val PPL throughout training (unusual pattern indicating strong generalization).

## Phase Transition Analysis (Steps 3K-5K)

### Discovery

At steps 3K-5K, both train and val PPL curves moved together - a rare synchronization event suggesting a fundamental shift in model behavior.

### Memory Matrix (M_init) Evolution

| Metric | Step 3K | Step 4K | Step 5K | Interpretation |
|--------|---------|---------|---------|----------------|
| Total Norm | 1.22 | 1.31 | 1.37 | Stronger representations |
| Avg Effective Rank | 328 | 318 | 312 | More specialized features |
| Top-1 Concentration | ~8% | ~9% | ~10% | Dominant directions emerging |

### Layer-wise Changes (3K→4K)

The transition from step 3K to 4K showed massive restructuring in deeper layers:

| Layer | Relative Change | Effective Rank Delta |
|-------|-----------------|---------------------|
| Layer 0 | 28.77% | -7.28 |
| Layer 1 | 28.54% | -10.41 |
| Layer 2 | 36.28% | -9.99 |
| Layer 3 | 43.00% | -5.82 |
| Layer 4 | **52.43%** | **-13.65** |
| Layer 5 | **53.52%** | **-11.56** |

### Interpretation

1. **Before (3K)**: Memory matrices diffuse, exploring representational space broadly
2. **Transition (3K-4K)**: Massive restructuring, especially in layers 4-5
3. **After (4K-5K)**: Stabilization around discovered patterns

This represents a **phase transition** from exploration to exploitation. The model discovered generalizable structure rather than memorizing.

### Supporting Evidence

- **M3 Mixer**: Alpha stable at ~0.50 (balanced local/global memory)
- **QK Circuit Breaker**: Zero saturation (no instability)
- **Polynomial Coefficients**: Fixed at [1.0, 1.0, 0.5]

## Anomalies to Monitor

1. **Train PPL > Val PPL**: Counter to typical overfitting pattern. Monitor for sustained delta.
2. **Deep layer activity**: Layers 4-5 showed most change during transition. Watch for continued specialization.

## Next Analysis Checkpoints

- [ ] Step 50K analysis (quarter mark)
- [ ] Step 100K analysis (halfway)
- [ ] Step 150K analysis (three-quarter mark)
- [ ] Final model analysis

---

## Failure Analysis (2026-01-21)

### Committee Review Feedback

External review identified critical concerns:
1. Val PPL < Train PPL inversion too early - suspicious
2. Zero saturation could mean dead gates, not healthy gates
3. Phase transition might be seed rejection, not healthy learning

### Diagnostic Results

**AUDIT 1: Tanh Gate Input Magnitudes - FAILED**

All layers show input magnitudes ~0.017, far below the 0.1 threshold for nonlinear operation:

| Layer | Mean \|x\| | % > 0.5 | Status |
|-------|-----------|---------|--------|
| 0 | 0.0173 | 0.0% | DEAD |
| 1 | 0.0176 | 0.0% | DEAD |
| 2 | 0.0174 | 0.0% | DEAD |
| 3 | 0.0178 | 0.0% | DEAD |
| 4 | 0.0182 | 0.0% | DEAD |
| 5 | 0.0176 | 0.0% | DEAD |

**AUDIT 2: Memory Mode Impact - NO EFFECT**

```
Mode 1 (NEVER reset):       PPL = 751.60
Mode 2 (reset every SHARD): PPL = 752.58
Mode 3 (reset every BATCH): PPL = 751.55

Difference: ~1 PPL (negligible)
```

Memory accumulation has no effect on model predictions.

**AUDIT 3: Baton Pass - PASSED**

First-token loss penalty of +11.3% is within acceptable range.

### Root Cause

The `input_gain` parameter in QK projection initializes to ~0.021 (`0.5/sqrt(576)`). Combined with normalized inputs, values reaching the tanh are ~0.017 - deep in the linear region where tanh(x) ≈ x.

The model found an equilibrium where the gating mechanism is completely bypassed. It's effectively a standard transformer.

### Implications

1. The "strong generalization" (val < train) was an artifact, not real
2. The phase transition was likely seed rejection
3. The memory system provides no benefit in current form
4. Loss improvements came from the transformer backbone, not memory

### Next Steps

See [PRD_RUN2_FIXES.md](../PRD_RUN2_FIXES.md) for required changes before Run #2.

---

*Analysis generated: 2026-01-21*
*Failure analysis added: 2026-01-21*
