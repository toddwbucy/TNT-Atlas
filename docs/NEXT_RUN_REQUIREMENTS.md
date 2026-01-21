# Next Training Run Requirements

**Status**: PLANNED
**Target**: Run #2 (different dataset, same architecture for comparison)

## Training Hooks Needed

The following metrics need to be logged DURING training (not extractable from checkpoints):

### Priority 1: Gradient Health

```python
# Add to training loop after loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        log_metric(f"grad_norm/{name}", grad_norm, step)
```

**Metrics to log:**
- `grad_norm/blocks.{i}.M_init` - Memory matrix gradients
- `grad_norm/blocks.{i}.attention.*` - Attention gradients
- `grad_norm/embedding` - Embedding layer gradient
- `grad_norm_total` - Global gradient norm (already have from grad_clip)

### Priority 2: Attention Dynamics

```python
# Add hook to attention forward pass
def attention_hook(module, input, output):
    # Compute attention entropy
    attn_weights = output[1]  # Assuming (output, attn_weights)
    entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(-1).mean()
    log_metric(f"attention_entropy/layer_{layer_idx}", entropy.item(), step)

    # Attention sparsity (% of weights > threshold)
    sparsity = (attn_weights > 0.1).float().mean()
    log_metric(f"attention_sparsity/layer_{layer_idx}", sparsity.item(), step)
```

**Metrics to log:**
- `attention_entropy/layer_{i}` - How spread out attention is
- `attention_sparsity/layer_{i}` - Concentration of attention

### Priority 3: Memory System Dynamics

```python
# Add to AtlasBlock forward pass
def memory_hook(module, input, output):
    # Memory read magnitude
    read_norm = memory_output.norm().item()
    log_metric(f"memory_read_norm/layer_{layer_idx}", read_norm, step)

    # Memory write magnitude (if tracking M updates)
    write_norm = delta_M.norm().item()
    log_metric(f"memory_write_norm/layer_{layer_idx}", write_norm, step)

    # Read/write ratio
    rw_ratio = read_norm / (write_norm + 1e-10)
    log_metric(f"memory_rw_ratio/layer_{layer_idx}", rw_ratio, step)
```

**Metrics to log:**
- `memory_read_norm/layer_{i}` - How much is being retrieved
- `memory_write_norm/layer_{i}` - How much is being stored
- `memory_rw_ratio/layer_{i}` - Balance of retrieval vs storage

### Priority 4: M3 Mixer Live State

```python
# Already have alpha in checkpoints, but live logging is better
# Add to M3Mixer forward pass
log_metric(f"m3_alpha/layer_{layer_idx}", self.get_alpha().item(), step)
log_metric(f"m3_alpha_grad/layer_{layer_idx}", self.alpha_logit.grad.item(), step)
```

## Logging Infrastructure

### Option A: JSONL (Simple, Current Approach)

```python
# metrics.jsonl - append per step
{"step": 1000, "grad_norm/blocks.0.M_init": 0.0023, "attention_entropy/layer_0": 2.31, ...}
```

**Pros**: Simple, works with current dashboard
**Cons**: Large files, slow to query

### Option B: SQLite (Better for Analysis)

```python
import sqlite3

# Schema
CREATE TABLE metrics (
    step INTEGER,
    name TEXT,
    value REAL,
    timestamp TEXT,
    PRIMARY KEY (step, name)
);

# Query example
SELECT step, value FROM metrics WHERE name = 'grad_norm/blocks.0.M_init';
```

**Pros**: Fast queries, easy aggregation
**Cons**: More setup

### Recommendation: SQLite + JSONL Summary

- SQLite for all metrics (fast queries)
- JSONL summary every N steps (compatibility with current tools)

## Implementation Checklist

- [ ] Add `src/metrics_logger.py` with SQLite backend
- [ ] Add gradient hooks to `scripts/train.py`
- [ ] Add attention hooks to `src/attention.py`
- [ ] Add memory hooks to `src/atlas_block.py`
- [ ] Update dashboard to read from SQLite
- [ ] Add variance band charts for new metrics
- [ ] Test with short run before full training

## Comparison Framework

For Run #2, we want to compare against Run #1:

| Metric | Run #1 Baseline | Run #2 Target |
|--------|-----------------|---------------|
| Val PPL @ 5K steps | 970.9 | Compare |
| Val PPL @ 10K steps | 827.1 | Compare |
| Phase transition step | ~3K-4K | Identify |
| Effective rank trend | 328 → 312 | Compare |
| Train > Val PPL | Yes (unusual) | Verify pattern |

## Dataset Candidates for Run #2

1. **Different domain**: Code (The Stack), scientific (arxiv)
2. **Different quality**: Raw web vs curated
3. **Different size**: Same 100M tokens for fair comparison

---

*Document created: 2026-01-21*
*Last updated: 2026-01-21*
