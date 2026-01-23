# TNT-Atlas: Memory-Augmented Transformer Training

> **WORK IN PROGRESS**: This codebase is under active development and has not yet produced a successfully trained model. The architecture and training methodology are experimental. The iterative failure analysis documented here is part of the research process.

## Overview

TNT-Atlas is an experimental implementation combining:

- **ATLAS**: Memory-Augmented Gating architecture with polynomial feature expansion
- **TNT (Transformer-in-Transformer)**: Hierarchical training methodology with shard-based context windows
- **Titans-inspired**: Hybrid caching and M3 mixing for memory state management

This project explores whether memory-augmented attention mechanisms can improve language model performance on long-context tasks.

## Current Status

### Run History

| Run | Gain | Result | Issue |
|-----|------|--------|-------|
| Run #1 | 0.5 | FAILED | Dead gates - tanh operating in linear region |
| Run #2 | 2.0 | FAILED | Linear gates - model bypassed memory system |
| Run #3 | 4.0 | FAILED | Gates still linear at low activations |
| Run #4 | 5.0 | FAILED | Gain parameter stuck (BF16 precision issue) |
| Run #5 | 6.0 | FAILED | Same BF16 precision issue |
| Run #6 | 5.0 + 100x LR | FAILED | Confirmed BF16 quantizing updates to zero |
| **Run #7** | **log-param** | **RUNNING** | ✓ Fix verified - gain learning correctly |

### Run #7 Progress (Current)

**Status:** Training in progress with BF16 precision fix (log-parameterization)

#### Training Metrics

| Step | Train Loss | Train PPL | Tokens | Progress |
|------|------------|-----------|--------|----------|
| 0 | 11.19 | 72,403 | 0 | 0% |
| 1000 | 7.51 | 1,826 | 512K | 0.5% |
| 4000 | 7.08 | 1,188 | 2.0M | 2.0% |
| 8000 | 6.64 | 765 | 4.1M | 4.1% |
| 12000 | 6.56 | 710 | 6.1M | 6.1% |
| 13500+ | ~6.7 | ~810 | ~6.9M | ~7% |

#### Validation Metrics

| Step | Train Loss | Val Loss | Val PPL | Train/Val Gap |
|------|------------|----------|---------|---------------|
| 11000 | 5.69 | 6.74 | 848 | 18.5% |
| 12000 | 6.56 | 6.72 | **827** | 2.4% |

#### Health Status
- ✅ Loss reduced 99% (72,403 → 827 PPL)
- ✅ Healthy train/val gap (no overfitting)
- ✅ All saturation levels at 0.0
- ✅ Effective gain stable at ~1.0
- ✅ No kill switches triggered
- ✅ Training speed stable at ~236 tok/s

**Key Discovery (BN-001):** BFloat16 precision at value 5.0 (~0.04) couldn't detect gradient updates of ~1e-5. Solution: log-parameterization keeps learnable parameter near 1.6 where precision is ~0.013. See [docs/build-notes.md](docs/build-notes.md).

**Architectural Insight:** Model learned to reduce effective gain from 5.0 → 1.0, suggesting optimal gain is near 1.0 for this architecture.

## Architecture

```
ATLAS-MAG BLOCK (with TNT training)
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Input x                                                   │
│       │                                                     │
│       ├──────────────────────────┐                          │
│       │                          │                          │
│       ▼                          ▼                          │
│  ┌─────────────────┐   ┌───────────────────────────┐       │
│  │ SLIDING WINDOW  │   │ ATLAS DEEP MEMORY         │       │
│  │ ATTENTION       │   │                           │       │
│  │                 │   │ • Q-K Projection (tanh)   │       │
│  │ • Window size w │   │ • Polynomial features φ   │       │
│  │ • Persistent P  │   │ • Omega rule (context c)  │       │
│  │ • Causal window │   │ • Newton-Schulz update    │       │
│  └────────┬────────┘   └─────────────┬─────────────┘       │
│           │                          │                      │
│           ▼                          ▼                      │
│      [LayerNorm]                [LayerNorm]                 │
│           │                          │                      │
│           ▼                          │                      │
│      [sigmoid]                       │                      │
│           │                          │                      │
│           └──────────► (*) ◄─────────┘                      │
│                         │                                   │
│                         ▼                                   │
│                  [Out Projection]                           │
│                         │                                   │
│                        (+) <── residual                     │
│                         │                                   │
│                         ▼                                   │
│                      Output                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
atlas_tnt/
├── configs/           # Pydantic-validated configuration
├── docs/              # Design documents and PRDs
├── scripts/           # Training and evaluation scripts
│   ├── train.py       # Main training loop with kill switches
│   ├── preflight_gates.py  # Pre-training validation
│   └── prepare_data.py     # Data preparation
├── src/               # Core implementation
│   ├── atlas_block.py      # AtlasMAG block
│   ├── qk_projection.py    # Q-K projection with tanh gates
│   ├── m3_mixing.py        # Memory state mixing
│   ├── polynomial.py       # Polynomial feature expansion
│   └── config_schema.py    # Pydantic V2 configuration
└── tests/             # Isolation and integration tests
```

## Key Design Decisions

### 1. No Hidden Defaults
All configuration uses Pydantic V2 with explicit required fields. If a config value is missing, the script fails fast with a clear error.

### 2. Kill Switches Over Monitoring
Training includes hard kill switches that terminate runs when architectural failures are detected:
- **Activation Kill Switch**: Terminates if tanh gates stay in linear region
- **Alpha Guardrails**: Terminates if M3 mixing collapses to extremes
- **Saturation Kill Switch**: Terminates if gates enter "coma" (saturated) state

### 3. Pre-Flight Gates
Mandatory validation before training:
- Gate 1: Proof of Life (gates can reach nonlinear operation)
- Gate 2: Baton Pass (state transfer across shard boundaries)
- Gate 3: Seed Competition (seeded model outperforms random)
- Gate 4: Memory Ablation (memory affects predictions)

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run pre-flight gates (MANDATORY)
python scripts/preflight_gates.py

# Run isolation suite
python tests/test_isolation.py

# Start training
python scripts/train.py --config configs/default.yaml
```

## References

This implementation draws from:
- ATLAS architecture papers
- TNT (Transformer-in-Transformer) training methodology
- Titans hybrid memory architectures
- Nested Learning (Ad Hoc Level Stacking)

## Disclaimer

**This code represents experimental research in progress.** No guarantees are made about:
- Correctness of implementation
- Training stability
- Model performance
- Production readiness

The documented failures and iterative fixes are an intentional part of the research methodology.

## License

This project is for research and educational purposes.
