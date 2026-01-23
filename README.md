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
| Run #7 | log-param | FAILED | Kill switch at step 16,600 - double normalization squeeze |
| **Run #8** | **v4.5 fix** | **PREPARING** | Query L2 norm removed (Issue #5, PR #6) |

### Run #7 Post-Mortem

**Status:** Kill switch activated at step 16,600

#### What Happened
- BF16 precision fix (BN-001) worked correctly - gain was learning
- Kill switch triggered: all 6 layers had activations < 0.1 for 1000 consecutive steps
- Root cause: **Double normalization squeeze** (Issue #5)

#### Root Cause Analysis

The architecture had two normalizations that compounded to squeeze the signal:

1. **Query L2 normalized**: `||q|| = 1.0`
2. **P Frobenius normalized**: `||P||_F = 1.0`, but `||P||_op ≈ 0.18`

Result: `||P_norm @ q|| ≈ 0.04` → tanh operates in linear region → dead gates

The model compensated by:
- Dropping gain from 5.0 → 1.0 (couldn't help - input already tiny)
- Increasing output_scale to 2-6 (trying to amplify weak signal)

But memory signal was still too weak to compete with full-strength attention.

#### Training Metrics (before kill switch)

| Step | Train Loss | Val Loss | Val PPL |
|------|------------|----------|---------|
| 11000 | 5.69 | 6.74 | 848 |
| 12000 | 6.56 | 6.72 | 827 |
| 16600 | ~6.8 | - | - |

### Run #8 Preparation (Current)

**Status:** Preparing to start

#### v4.5 Fix (PR #6)

Removed query L2 normalization while keeping key L2 normalization:

| Configuration | tanh input (gain=5.0) | Status |
|---------------|----------------------|--------|
| Before (v4.4) | 0.007 | ❌ Linear region (dead gates) |
| After (v4.5)  | 0.18  | ✅ Nonlinear region (functional) |

Changes:
- `atlas_block.py`: Query projection no longer L2-normalized
- `minimal_block.py`: Aligned test block with production behavior
- `qk_projection.py`: Documented v4.5 coordination

#### Expected Outcomes
- Tanh gates should operate in nonlinear region from the start
- Learnable gain has room to tune up or down
- Memory pathway should compete effectively with attention
- Kill switch should NOT trigger

**Key Discovery (BN-001):** BFloat16 precision at value 5.0 (~0.04) couldn't detect gradient updates of ~1e-5. Solution: log-parameterization keeps learnable parameter near 1.6 where precision is ~0.013. See [docs/build-notes.md](docs/build-notes.md).

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
