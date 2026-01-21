# Testing Strategy - Atlas TNT

This document defines the testing strategy for the Atlas TNT training codebase.

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Test individual components in isolation with mocked dependencies.

| Component | File | Tests |
|-----------|------|-------|
| Config Schema | `test_config_schema.py` | Pydantic validation, required fields, type coercion |
| Data Loader | `test_data_loader.py` | Chunking, shard boundaries, collation |
| Polynomial | `test_polynomial.py` | Feature expansion, gradient flow |
| QK Projection | `test_qk_projection.py` | Gain computation, circuit breaker |
| M3 Mixing | `test_m3_mixing.py` | Alpha update, state blending |
| Cache Manager | `test_cache_manager.py` | Staleness detection, invalidation |
| Newton-Schulz | `test_newton_schulz.py` | Orthonormalization, convergence |

**Criteria:**
- Fast (<1s per test)
- No GPU required
- No external dependencies (HuggingFace, etc.)
- 100% coverage target for critical paths

### 2. Integration Tests (`tests/integration/`)

Test component interactions and data flow.

| Test | File | What It Tests |
|------|------|---------------|
| Model Forward | `test_model_forward.py` | Full forward pass through AtlasModel |
| Training Step | `test_training_step.py` | Forward + backward + optimizer step |
| Checkpoint Roundtrip | `test_checkpoint.py` | Save → Load → Resume |
| Data Pipeline | `test_data_pipeline.py` | Tokenization → DataLoader → Batches |

**Criteria:**
- Can be slow (up to 30s per test)
- May require GPU (marked with `@pytest.mark.gpu`)
- May require network (marked with `@pytest.mark.network`)

### 3. Type Safety (`mypy`)

Static type checking for all Python code.

```bash
# Run type checking
mypy src/ scripts/ --strict
```

**Configuration:** `pyproject.toml`

**Criteria:**
- Zero type errors in CI
- All functions have type hints
- No `Any` types in public interfaces

### 4. Isolation Tests (`tests/test_isolation.py`)

**Already exists.** Component stress tests that validate ML behavior.

| Test | Purpose |
|------|---------|
| Linear Line | Polynomial layer can fit basic patterns |
| Compass Recovery | M3 mixing allows recovery from poisoned state |
| Cache Staleness | Version assertion rejects stale cache |
| Boundary Stress | Rapid shard crossings don't explode |
| Polynomial Capacity | Polynomial layer is numerically stable |
| Tanh→NS Interaction | Saturation doesn't break Newton-Schulz |
| Diff LR→M3 Interaction | Extreme LR ratios don't break M3 |
| Cache→M3 Interaction | Staleness detected before M3 mixing |

**Criteria:**
- ALL 8 tests MUST pass before training
- Run automatically before training starts

### 5. Scientific Validation Tests (`tests/test_scientific_validation.py`)

**Committee-required tests** validating theoretical claims from NL paper.

| Test Class | Tests | Purpose |
|------------|-------|---------|
| `TestInitializationEfficacy` | 3 | Validates NL Section 7.3 (seed vs random init) |
| `TestSpectralInterference` | 3 | Validates frequency isolation (hold and adapt) |
| `TestRetrievalFidelity` | 3 | Validates QK projection alignment |
| `TestContextContinuity` | 3 | Validates M3 baton pass at shard boundaries |

**Test Details:**

**Initialization Efficacy (Seed vs Random)**
- `test_seed_produces_different_init` - Verifies seed creates different M_init
- `test_seed_produces_lower_initial_loss` - Seed model should not be worse
- `test_seed_enables_faster_gradient_descent` - Both should learn

**Spectral Interference (Hold and Adapt)**
- `test_differential_lr_creates_frequency_separation` - Global changes slowly
- `test_m3_mixing_preserves_global_information` - Boundary mixing works
- `test_high_low_frequency_pattern_separation` - Memory responds to patterns

**Retrieval Fidelity (QK Projection Alignment)**
- `test_qk_projection_similarity_preservation` - Similar inputs → similar outputs
- `test_qk_attention_retrieval_accuracy` - Information preserved in attention
- `test_circuit_breaker_prevents_gain_explosion` - Tanh gate bounds output

**Context Continuity (Baton Pass)**
- `test_m3_state_inheritance` - M3 mixing blends prev and init states
- `test_information_persists_across_boundary` - State inheritance works
- `test_baton_pass_maintains_learning_momentum` - No catastrophic forgetting

**Criteria:**
- ALL 12 tests MUST pass before training
- Run with: `pytest tests/test_scientific_validation.py -v`

### 6. Training Script Tests (`tests/scripts/`)

Smoke tests for CLI scripts.

| Script | Test | What It Validates |
|--------|------|-------------------|
| `train.py` | `test_train_smoke.py` | Config loading, model creation, 1 step |
| `prepare_data.py` | `test_prepare_data_smoke.py` | Tokenization, output files created |
| `evaluate.py` | `test_evaluate_smoke.py` | Checkpoint loading, validation pass |

**Criteria:**
- Use tiny test fixtures (not real data)
- Validate script runs without error
- Validate output files are created correctly

## Test Markers

```python
@pytest.mark.unit          # Fast unit tests (no GPU, no network)
@pytest.mark.integration   # Integration tests (may be slow)
@pytest.mark.gpu           # Requires CUDA GPU
@pytest.mark.network       # Requires network access
@pytest.mark.slow          # Takes >5s to run
@pytest.mark.isolation     # ML isolation tests (run before training)
```

## Running Tests

```bash
# All tests (84 total)
pytest

# Unit tests only (fast, no GPU)
pytest -m unit

# Integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests (for CPU-only machines)
pytest -m "not gpu"

# Isolation suite only (8 tests)
pytest tests/test_isolation.py

# Scientific validation suite only (12 tests)
pytest tests/test_scientific_validation.py -v

# Both isolation suites (20 tests)
pytest tests/test_isolation.py tests/test_scientific_validation.py -v

# With coverage
pytest --cov=src --cov-report=html

# Type checking
mypy src/ scripts/ --strict
```

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py                     # Shared fixtures (tiny_config, etc.)
├── test_isolation.py               # 8 ML isolation tests
├── test_scientific_validation.py   # 12 scientific validation tests (NL paper)
│
├── unit/                           # Unit tests
│   ├── __init__.py
│   ├── test_config_schema.py       # Pydantic validation, required fields
│   ├── test_data_loader.py         # Chunking, shard boundaries, collation
│   ├── test_polynomial.py          # Feature expansion, gradient flow
│   ├── test_qk_projection.py       # Gain computation, circuit breaker
│   ├── test_m3_mixing.py           # Alpha update, state blending
│   ├── test_cache_manager.py       # Staleness detection, invalidation
│   └── test_newton_schulz.py       # Orthonormalization, convergence
│
├── integration/                    # Integration tests
│   ├── __init__.py
│   ├── test_model_forward.py       # Full forward pass
│   ├── test_training_step.py       # Forward + backward + optimizer
│   ├── test_checkpoint.py          # Save → Load → Resume
│   └── test_data_pipeline.py       # Tokenization → DataLoader
│
├── scripts/                        # Script smoke tests
│   ├── __init__.py
│   ├── test_train_smoke.py
│   ├── test_prepare_data_smoke.py
│   └── test_evaluate_smoke.py
│
└── fixtures/                       # Test data fixtures
    ├── tiny_config.yaml            # Minimal valid config
    ├── tiny_tokens.npy             # Small tokenized dataset
    └── tiny_checkpoint.pt          # Small checkpoint for testing
```

## CI Pipeline

```yaml
# .github/workflows/test.yml
jobs:
  test:
    steps:
      - name: Type Check
        run: mypy src/ scripts/ --strict

      - name: Unit Tests
        run: pytest -m unit --cov=src

      - name: Integration Tests
        run: pytest -m integration

      - name: Isolation Suite
        run: pytest tests/test_isolation.py

      - name: Scientific Validation Suite
        run: pytest tests/test_scientific_validation.py -v

      - name: All Tests
        run: pytest tests/ -v
```

## Test Fixtures

### `tiny_config.yaml`
Minimal config with small dimensions for fast testing:
- `d_model: 32` (instead of 512)
- `n_layers: 2` (instead of 6)
- `batch_size: 2`

### `tiny_tokens.npy`
Small pre-tokenized dataset:
- ~10K tokens
- Same format as production data

### `tiny_checkpoint.pt`
Small checkpoint for load/resume testing:
- Uses tiny config
- Step 10, random weights

## Coverage Targets

| Category | Target |
|----------|--------|
| `src/config_schema.py` | 100% |
| `src/data_loader.py` | 90% |
| `src/seed_model_loader.py` | 80% |
| `src/*.py` (core) | 80% |
| `scripts/*.py` | 70% |

## Adding New Tests

1. **Unit Test**: Add to `tests/unit/test_<module>.py`
2. **Integration Test**: Add to `tests/integration/`
3. **Mark appropriately**: `@pytest.mark.unit`, etc.
4. **Update this doc** if adding new categories

## Pre-Training Checklist

Before allocating GPU compute:

- [ ] `mypy src/ scripts/ --strict` passes with 0 errors
- [ ] `pytest -m unit` all passing
- [ ] `pytest tests/test_isolation.py` all 8 passing
- [ ] `pytest tests/test_scientific_validation.py` all 12 passing
- [ ] `pytest tests/` all 84 tests passing
- [ ] Config validated: `python -c "from src.config_schema import load_config; load_config('configs/default.yaml')"`
- [ ] Seed model config verified (if enabled): `seed_model.enabled: true`

## Test Counts Summary

| Category | Count | Command |
|----------|-------|---------|
| Unit tests | 54 | `pytest -m unit` |
| Integration tests | 9 | `pytest -m integration` |
| Isolation tests | 8 | `pytest tests/test_isolation.py` |
| Scientific validation | 12 | `pytest tests/test_scientific_validation.py` |
| **Total** | **84** | `pytest tests/` |
