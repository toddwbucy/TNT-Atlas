# Atlas TNT Build Notes

Operational findings and lessons learned during Atlas development. These notes capture critical discoveries that inform architecture decisions, especially for production deployment at scale.

---

## BN-001: Low-Precision Parameter Learning Failure

**Date:** 2025-01-22
**Severity:** Critical
**Affects:** BFloat16, Float16, FP8 training
**Component:** QKProjectionLayer.input_gain

### Problem

Learnable `input_gain` parameter initialized at 5.0 failed to learn despite:
- Being correctly included in optimizer
- Having 100x learning rate boost
- Gradients flowing (verified via exp_avg in optimizer state)

The parameter remained **exactly 5.000000000000** after 20,000+ training steps.

Meanwhile, `output_scale` (same optimizer group, same LR) learned successfully, moving from 1.0 to values like 5.04, 2.87, 3.35, etc.

### Root Cause

**BFloat16 precision limits at large parameter values.**

BFloat16 has only 7-8 bits of mantissa. The smallest representable change at a given value is approximately:

```
resolution ≈ value × 2^(-7) ≈ value × 0.0078
```

| Parameter Value | BF16 Resolution | FP8 (E4M3) Resolution |
|-----------------|-----------------|----------------------|
| 1.0             | ~0.008          | ~0.125               |
| 5.0             | ~0.04           | ~0.625               |
| 10.0            | ~0.08           | ~1.25                |

**Observed behavior:**
```python
# BFloat16 precision test
x = torch.tensor(5.0, dtype=torch.bfloat16)
x + 0.001  # = 5.0 (update lost!)
x + 0.01   # = 5.0 (update lost!)
x + 0.1    # = 5.1 (works)
```

The gradient updates to `input_gain` were ~1e-5 to 1e-7 per step, which is **1000-10000x below** the precision threshold at value 5.0. All updates were quantized to zero.

`output_scale` started at 1.0 where BF16 has ~5x better relative precision, AND had larger gradients due to its position after tanh (more direct path to loss).

### Why This Matters for FP8

FP8 (E4M3 format) has only 3 bits of mantissa:
- At value 5.0, resolution is ~0.625
- Parameters would need updates >0.3 to register ANY change
- This will affect many more parameters than just gain

Understanding this at BF16 is critical preparation for FP8 deployment.

### Design Principles for Low-Precision Robustness

1. **Keep learnable parameters near zero**
   - Precision is best near 0 (resolution ~= 2^-mantissa_bits)
   - Use `base_value + learnable_delta` where delta starts at 0
   - Similar to LoRA's design philosophy

2. **Log-parameterization for multiplicative factors**
   - Instead of learning `gain=5.0`, learn `log_gain=1.6094`
   - Forward pass: `effective_gain = exp(log_gain)`
   - Parameter stays near 1.6 where BF16 precision is ~0.013 (3x better)

3. **Avoid large initial values for learnable params**
   - If you need output scale of 5.0, use `fixed_scale * (1 + learnable_delta)`
   - `learnable_delta` stays near 0, `fixed_scale` is constant

4. **Consider gradient magnitude vs precision threshold**
   - If `gradient × learning_rate < precision_threshold`, param won't learn
   - May need higher LR, gradient scaling, or reparameterization

### Recommended Fix

**Before (fails in BF16):**
```python
# input_gain starts at 5.0, can't learn
self.input_gain = nn.Parameter(torch.ones(d_model) * 5.0)

# Forward:
q_gated = self.input_gain * q_aligned
```

**After (low-precision robust):**
```python
# Log-parameterization: parameter stays near 1.6
self.log_gain = nn.Parameter(torch.ones(d_model) * 1.6094)  # log(5.0)

# Forward:
effective_gain = torch.exp(self.log_gain)  # = 5.0 at init
q_gated = effective_gain * q_aligned
```

**Alternative (additive delta):**
```python
# Fixed base + learnable delta near zero
self.gain_base = 5.0  # constant, not learned
self.gain_delta = nn.Parameter(torch.zeros(d_model))  # starts at 0

# Forward:
effective_gain = self.gain_base + self.gain_delta
q_gated = effective_gain * q_aligned
```

### Verification Checklist

When adding learnable parameters, verify:
- [ ] Initial value allows sufficient precision for expected gradient magnitude
- [ ] `gradient × lr > value × 2^(-mantissa_bits)` for target precision
- [ ] Consider log-parameterization for multiplicative factors
- [ ] Test that parameter actually changes during training (check with high precision)

### Detection Method

To detect this issue in other parameters:
```python
# Check if parameter is stuck at init value
param = model.some_param
init_value = param.data.clone()
# ... train for N steps ...
diff = (param.data - init_value).abs().max()
if diff < init_value.abs().max() * 1e-6:
    print(f"WARNING: Parameter may be stuck due to precision limits")
```

### References

- Run #4, #5: Gains frozen at 5.0/6.0 for 20K steps
- Run #6: Confirmed with 100x LR boost, still frozen
- Checkpoint analysis: `runs_run6/checkpoints/step_00001000.pt`

---

## Template for Future Notes

```markdown
## BN-XXX: Title

**Date:** YYYY-MM-DD
**Severity:** Critical/High/Medium/Low
**Affects:** Components/systems affected
**Component:** Specific code location

### Problem
What was observed

### Root Cause
Why it happened

### Design Principles
General lessons learned

### Recommended Fix
Specific code changes

### Verification Checklist
How to prevent recurrence

### References
Related runs, files, experiments
```
