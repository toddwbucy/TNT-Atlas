# PRD: Atlas Training Runs #2 & #3 - Addressing Architectural Failures

**Status**: ACTIVE (Rev 5 - Run #2 Results, Run #3 Fixes)
**Created**: 2026-01-21
**Updated**: 2026-01-21
**Run #1 Status**: FAILED (dead gates, gain=0.5)
**Run #2 Status**: FAILED (linear gates, gain=2.0 - gates bypassed)
**Run #3 Status**: PENDING (gain=4.0)

---

## Executive Summary

Run #1 (atlas_58m_fineweb_100M) revealed critical architectural failures in the memory system. Despite appearing to train (loss decreasing, PPL improving), diagnostic analysis shows the core memory-augmented gating mechanism is **completely non-functional**. The model is effectively operating as a standard transformer without memory augmentation.

This PRD documents the failures, root causes, and required fixes before starting Run #2.

### Committee Feedback Summary (Rev 2)

External review identified that our original fixes were **too passive**:

> "They're trying to monitor their way out of a problem that requires enforcement."

Key criticisms:
1. **Monitoring ≠ Enforcement**: Alerts that humans can ignore are worthless at 3am
2. **Seed Rejection Hypothesis**: The phase transition was the model killing the memory, not learning to use it
3. **Baton Pass Penalty Too High**: 11.3% is a failure, not "acceptable"
4. **Unit Tests Don't Catch Emergent Behaviors**: The system failed while all unit tests passed

**New Mandate**: All validation checks become **hard kill switches**, not passive monitoring.

### Second Committee Feedback Summary (Rev 4)

After running pre-flight tests and seed competition, committee raised additional concerns:

> "Dead gates is the 'how', not the 'why'. The 'why' is seed rejection. And cranking the gain to 2.0 might just make the rejection FASTER - louder noise gets rejected more violently."

Key criticisms:
1. **Gain=2.0 Risk**: Higher gain = louder noise = potentially faster rejection
2. **Seed Competition Too Permissive**: "Not worse than random" is meaningless - seed must be BETTER
3. **Need Gradient Health Check**: Prove M_init is receiving gradients (organ is alive, not just present)
4. **Baton Pass 5% Still Too High**: Drop to 2% or measure recovery slope
5. **M3 Alpha Frequency Collapse Risk**: Alpha=0.5 could pin to 0 or 1, need orthogonality check
6. **TNT + High Gain Boundary Risk**: Potential gradient explosion at shard boundaries
7. **Pre-Step-1 Continuity Check**: Verify memory works BEFORE training starts

**New Mandate**: Tighten all thresholds. Add gradient health checks. Stress test boundaries.

---

## Run #2 Results (FAILED)

**Duration**: ~3000 steps (~25 minutes)
**Loss**: 11.4 → 6.9 (model was learning)
**Gates**: Linear mode throughout (saturation=0.0)

### What Happened

Run #2 with gain=2.0 showed initial promise:
- Loss decreased steadily (good learning signal)
- M_init gradients healthy (memory receiving gradients)
- All kill switches passed
- Activations grew initially (0.003 → 0.05)

**But activations plateaued at ~0.05:**

| Step | Layer 0 | Layer 5 | Status |
|------|---------|---------|--------|
| 0 | 0.003 | 0.003 | Growing |
| 500 | 0.043 | 0.006 | Growing |
| 1000 | 0.050 | 0.034 | Plateau |
| 2000 | 0.075 | 0.075 | Peak |
| 3000 | 0.056 | 0.058 | **Dropped** |

### Why It Failed

1. **Activations plateaued at 50% of threshold**: Peak was 0.075, never reached 0.1
2. **Model learned to bypass gates**: Loss improved while gates stayed linear
3. **Saturation remained 0.0**: Gates never entered nonlinear regime
4. **Gates became oscillatory**: Instead of growing toward 0.1, they oscillated 0.04-0.07

### The Diagnosis

With gain=2.0 and activation magnitude ~0.05:
- Effective tanh input: `2.0 × 0.05 = 0.10`
- `tanh(0.10) ≈ 0.0997` → essentially linear (slope ≈ 0.99)

The model found it easier to route around the memory system than through it. The gates provided no meaningful nonlinear gating - they were expensive linear transforms.

**Committee was right**: "Cranking the gain to 2.0 might just make the rejection FASTER" - except the model didn't reject violently, it just ignored the gates entirely.

---

## Run #3 Fixes

### Fix 13: Increase Gain to 4.0 (MANDATORY)

**Rationale**: With activation ~0.05, we need gain=4.0 to get meaningful nonlinearity:
- Effective tanh input: `4.0 × 0.05 = 0.20`
- `tanh(0.20) ≈ 0.197` → 1.5% nonlinearity (still marginal)
- If activations grow to 0.1: `4.0 × 0.1 = 0.4` → `tanh(0.4) ≈ 0.38` → 5% nonlinearity

**Implementation**:
```python
# qk_projection.py
self.input_gain = nn.Parameter(torch.ones(d_model) * 4.0)
```

### Fix 14: Raise Circuit Breaker to 6.0

**Rationale**: With gain=4.0, the old circuit breaker (2.0) would immediately trip.

**Implementation**:
```python
# qk_projection.py
gain_circuit_breaker: float = 6.0  # was 2.0
```

### Risk Assessment for Run #3

**Potential Issues**:
1. **Higher gain = more noise**: If memory signal is still noisy, model may reject faster
2. **Gradient explosion risk**: Large gain × large activation = large gradients
3. **Saturation risk**: If activations grow AND gain is high, could enter coma zone

**Mitigations**:
1. Saturation kill switch still active (patience=100 steps at >20% saturation)
2. Gradient clipping at 1.0
3. Monitor for rapid loss spikes in first 100 steps

---

## Failure Analysis

### Failure #1: Dead Tanh Gates (CRITICAL)

**Symptom**: QK circuit breaker reported zero saturation (appeared healthy)

**Root Cause**: Tanh gate input magnitudes are ~0.017, far below the 0.1 threshold for nonlinear operation. The gates operate entirely in their linear region.

**Evidence**:
```
Layer    Mean |x|     % > 0.5    Status
Layer 0  0.0173       0.0%       DEAD (linear region)
Layer 1  0.0176       0.0%       DEAD (linear region)
Layer 2  0.0174       0.0%       DEAD (linear region)
Layer 3  0.0178       0.0%       DEAD (linear region)
Layer 4  0.0182       0.0%       DEAD (linear region)
Layer 5  0.0176       0.0%       DEAD (linear region)
```

**Impact**: The memory gating mechanism is bypassed. The O(d²) polynomial feature expansion and QK projection are just expensive linear transforms.

**Committee Analysis**: "If the saturation is literally zero, it implies the tanh gate is operating completely in its linear region. Meaning it's not really acting like a gate at all."

### Failure #2: Memory Has No Effect on Loss

**Symptom**: Validation PPL < Training PPL (appeared to be "strong generalization")

**Root Cause**: Memory accumulation has zero impact on model predictions.

**Evidence**:
```
Mode 1 (NEVER reset):       PPL = 751.60
Mode 2 (reset every SHARD): PPL = 752.58
Mode 3 (reset every BATCH): PPL = 751.55

Difference: ~1 PPL across ALL modes
```

**Impact**: The model treats every chunk independently. There is no effective long-range memory or context continuity.

**Committee Analysis**: "If the model just collapses when you freeze global memory, it means the global memory was basically a freeloader and the local memory was carrying the entire load."

### Failure #3: Validation Not Apples-to-Apples (Medium)

**Symptom**: Validation consistently outperformed training

**Root Cause**: Original eval script used `shard_boundary=False` always, never resetting memory.

**Evidence**: Code review found hardcoded `shard_boundary=False` in evaluate.py

**Impact**: Misleading metrics. However, since memory doesn't work anyway, this didn't actually affect results.

**Fix Applied**: Updated evaluate.py to respect shard boundaries. But the deeper issue remains.

### Failure #4: Phase Transition Was Seed Rejection (CRITICAL - Rev 2)

**Symptom**: 53% weight change in layers 4-5 during steps 3K-5K

**Original Interpretation**: "Healthy phase transition from exploration to exploitation"

**Actual Reality**: **Seed model rejection**. The model actively killed the memory system.

**Committee Analysis (Rev 2)**:
> "SmolLM is a competent, pre-trained transformer. It knows English. It knows grammar. Then they graft this super complex Atlas memory architecture on top. At step zero, the memory is just outputting garbage. So the model wakes up. It has a choice: use the clean, competent SmolLM path, or route information through this noisy, chaotic memory module. What does gradient descent do? It suppresses the noise. It slams the memory gates shut so the seed can do its job."

**The Organ Transplant Analogy**:
- SmolLM = healthy existing organ (competent transformer)
- Atlas memory = transplanted organ (complex, initially noisy)
- Phase transition = immune rejection (body killing the new organ)
- Current state = the body rejected the transplant

**Why This Matters**:
The massive weight spike wasn't the model learning to use memory - it was the model learning to **bypass** it. The SmolLM seed literally bullied the Atlas architecture into submission.

**Implication for Run #2**: Simply making gates "louder" won't work. If the memory is still noisy relative to the seed, the model will just work harder to shut it down again. The memory must be **mathematically attractive** from step 1.

---

## Root Cause Analysis

### Why Are the Tanh Gates Dead?

The QK projection uses learnable `input_gain` parameters to scale inputs before the tanh:

```python
scaled = x * self.input_gain  # input_gain initialized to 0.5/sqrt(d_model)
output = torch.tanh(scaled)
```

With `d_model=576`, initial gain is `0.5/24 ≈ 0.021`. Combined with normalized inputs, the scaled values are ~0.017 - deep in tanh's linear region.

**The gain never learned to increase** because:
1. With such small inputs, gradients through tanh are ~1.0 (linear)
2. There's no gradient signal telling the model to increase the gain
3. The model found an equilibrium where the gate is essentially bypassed

### Why Didn't the Safety Rails Catch This?

The PRD specified monitoring for saturation, but:
1. Zero saturation was interpreted as "healthy" (no instability)
2. The correct check should have been for **minimum** activation magnitude
3. There was no "liveness" check for the gates

---

## Required Fixes for Run #2

### Fix 1: Gate Liveness Check (MANDATORY)

Add to validation suite:

```python
def check_gate_liveness(model, test_input):
    """Verify tanh gates can actually saturate."""
    # Inject high-magnitude input
    test_scaled = test_input * 10.0  # Force large values

    # Check if any layer can produce |tanh(x)| > 0.9
    for layer in model.blocks:
        output = layer.qk_proj(test_scaled)
        max_activation = output.abs().max()
        if max_activation < 0.9:
            raise RuntimeError(f"Gate liveness check failed: max={max_activation}")
```

Run this at:
- Model initialization
- Every 1000 steps during training
- Before each validation run

### Fix 2: Input Gain Initialization (MANDATORY)

**⚠️ UPDATE: gain=2.0 FAILED in Run #2. See Fix 13 for gain=4.0.**

Change `input_gain` initialization to ensure nonlinear operation from the start:

```python
# Run #1 (broken): gain = 0.5/sqrt(d_model) ≈ 0.02 → DEAD gates
# Run #2 (failed): gain = 2.0 → LINEAR gates (activations plateaued at 0.05)
# Run #3 (current): gain = 4.0 → attempting nonlinear operation
self.input_gain = nn.Parameter(torch.ones(d_model) * 4.0)
```

Or use adaptive initialization based on input statistics:

```python
# Calibrate gain to achieve target activation magnitude
target_tanh_input = 1.0  # Ensures ~76% of tanh range is used
self.input_gain = nn.Parameter(torch.ones(d_model) * target_tanh_input / expected_input_std)
```

**⚠️ RISK NOTE (Rev 4 - Committee Concern):**

> "Dead gates is the 'how', not the 'why'. The 'why' is seed rejection. And cranking the gain to 2.0 might just make the rejection FASTER - louder noise gets rejected more violently."

**The Concern**: Higher gain = louder memory signal = potentially FASTER rejection by the seed model.

**Mitigation Strategy**:
1. Run **Gate 5 (Boundary Stress Test)** with gain=2.0 BEFORE committing
2. Monitor gradient norms closely in first 1000 steps
3. If seeded model shows MORE volatility than random at 100 steps, gain may be too high
4. Consider graduated gain schedule: start at 1.0, increase to 2.0 over first 1K steps

**Alternative Approach** (if gain=2.0 causes problems):
```python
# Graduated gain schedule
def get_gain_for_step(step, warmup_steps=1000):
    if step < warmup_steps:
        # Linear ramp from 1.0 to 2.0
        return 1.0 + (step / warmup_steps)
    return 2.0
```

This lets the seed model "get used to" the memory system before the gates become loud.

### Fix 3: Activation Kill Switch (MANDATORY - Rev 2)

**Changed from monitoring to enforcement per committee feedback.**

> "It's 3am. A training run has already cost 10 grand. The loss curve is still going down. You get a little alert that says 'low activation.' Are you going to be the one to pull the plug? You're going to rationalize it."

**Implementation**: Hard kill switch, no human in the loop.

```python
# In training loop - KILL SWITCH, not monitoring
class ActivationKillSwitch:
    def __init__(self, threshold=0.1, patience=1000):
        self.threshold = threshold
        self.patience = patience
        self.steps_below_threshold = 0

    def check(self, model, step):
        """Check activation magnitudes. TERMINATES if gates are dying."""
        for i, block in enumerate(model.blocks):
            magnitude = block.qk_proj_layer.last_input_magnitude.mean()
            log_metric(f'tanh_input_magnitude/layer_{i}', magnitude, step)

            if magnitude < self.threshold:
                self.steps_below_threshold += 1
                logger.warning(
                    f"Step {step}: Layer {i} activation {magnitude:.4f} < {self.threshold} "
                    f"({self.steps_below_threshold}/{self.patience} steps)"
                )

                if self.steps_below_threshold >= self.patience:
                    logger.error(
                        f"KILL SWITCH ACTIVATED: Activations below {self.threshold} "
                        f"for {self.patience} consecutive steps. Gates are dead."
                    )
                    save_diagnostic_checkpoint(model, step, reason="dead_gates")
                    sys.exit(1)  # NO DIALOGUE. NO "ARE YOU SURE". JUST DIE.
            else:
                self.steps_below_threshold = 0  # Reset counter
```

**Rationale**: A crash is infinitely better than a coma. A crash forces you to debug. A coma wastes your budget.

### Fix 4: Memory Ablation Kill Switch (MANDATORY - Rev 2)

**Changed from logging to auto-termination per committee feedback.**

> "You turn off the memory, and if the perplexity doesn't spike - I mean really spike - that's your proof. The script should auto-terminate right there. No zombies."

**Implementation**: Automated ablation with hard kill switch.

```python
def validate_memory_impact(model, val_data, step, min_impact=10.0):
    """
    Verify memory actually affects predictions.
    TERMINATES RUN if memory has no effect.
    """
    ppl_with_memory = evaluate(model, val_data, use_memory=True)
    ppl_without_memory = evaluate(model, val_data, use_memory=False)

    impact = ppl_without_memory - ppl_with_memory
    log_metric('memory_impact_ppl', impact, step)

    logger.info(
        f"Step {step} Memory Ablation: "
        f"with_memory={ppl_with_memory:.2f}, without={ppl_without_memory:.2f}, "
        f"impact={impact:+.2f} PPL"
    )

    if impact < min_impact:
        logger.error(
            f"KILL SWITCH ACTIVATED: Memory ablation shows only {impact:.1f} PPL impact "
            f"(threshold: {min_impact}). Memory system is not contributing."
        )
        save_diagnostic_checkpoint(model, step, reason="memory_ineffective")
        sys.exit(1)  # ZOMBIE DETECTED. KILL IT.

    return impact
```

**Schedule**: Run at steps 1K, 5K, 10K, 25K, 50K

**Threshold**: Must show ≥10 PPL impact (raised from 5 per committee feedback - "really spike")

### Fix 5: Baton Pass Gate (MANDATORY - Rev 4 TIGHTENED)

**Upgraded from RECOMMENDED to MANDATORY, converted to hard gate.**

> "If you're dropping the baton every 2,048 tokens, you're not running a marathon. You're running a series of disconnected sprints. The fact that they marked 11% as acceptable is terrifying."

**Threshold**: ≤**2%** first-token penalty (Rev 4 - tightened from 5%)

**Committee feedback (Rev 4)**: "5% is still too high. Drop to 2% or measure recovery slope."

```python
class BatonPassGate:
    """
    Go/No-Go gate for context continuity.
    Run BEFORE launching full training.
    """
    MAX_PENALTY = 0.02  # 2% max acceptable penalty (Rev 4 - tightened from 5%)

    def validate(self, model, val_data, n_shards=20):
        """
        Measure first-token loss penalty at shard boundaries.
        BLOCKS TRAINING if penalty exceeds threshold.
        """
        first_token_losses = []
        rest_token_losses = []

        memory_states = [None] * n_layers
        for shard_idx in range(n_shards):
            for chunk_idx in range(chunks_per_shard):
                shard_boundary = (chunk_idx == 0) and (shard_idx > 0)

                logits, memory_states, _ = model(
                    input_ids, memory_states=memory_states,
                    shard_boundary=shard_boundary
                )

                if shard_boundary:
                    token_losses = compute_per_token_loss(logits, labels)
                    first_token_losses.append(token_losses[:, 0].mean())
                    rest_token_losses.append(token_losses[:, 1:].mean())

        avg_first = np.mean(first_token_losses)
        avg_rest = np.mean(rest_token_losses)
        penalty = (avg_first / avg_rest) - 1.0

        logger.info(
            f"Baton Pass Gate: first_token={avg_first:.4f}, rest={avg_rest:.4f}, "
            f"penalty={penalty*100:+.1f}%"
        )

        if penalty > self.MAX_PENALTY:
            logger.error(
                f"BATON PASS GATE FAILED: {penalty*100:.1f}% penalty exceeds "
                f"{self.MAX_PENALTY*100}% threshold. Memory is not transferring context."
            )
            raise RuntimeError(
                f"Baton pass penalty {penalty*100:.1f}% > {self.MAX_PENALTY*100}%. "
                "Do not launch training. Debug the M3 mixing first."
            )

        logger.info(f"Baton Pass Gate: PASSED ({penalty*100:.1f}% <= {self.MAX_PENALTY*100}%)")
        return penalty
```

**When to Run**: Pre-flight check before ANY training run. If this fails, don't launch.

**Alternative Metric (Rev 4)**: If 2% seems unreachable, measure **recovery slope** instead:
- How many tokens until first-token penalty recovers to baseline?
- If recovery > 10 tokens, context isn't transferring properly
- This is a softer metric that allows brief spikes if recovery is fast

### Fix 6: Stress Test in Isolation Suite (RECOMMENDED)

Add to `tests/test_isolation.py`:

```python
def test_gate_can_saturate():
    """Verify QK gates can reach saturation with appropriate input."""
    model = create_test_model()

    # Generate high-magnitude input
    x = torch.randn(1, 128, 576) * 5.0

    for block in model.blocks:
        output = block.qk_proj_layer(x)
        assert output.abs().max() > 0.9, "Gate failed to saturate with large input"

def test_memory_mode_difference():
    """Verify memory modes produce different outputs."""
    model = create_test_model()

    # Same input, different memory modes
    x = torch.randn(1, 128, 576)

    out_fresh = model(x, memory_states=None)
    out_with_memory = model(x, memory_states=create_mock_memory())

    diff = (out_fresh - out_with_memory).abs().mean()
    assert diff > 0.01, f"Memory mode had no effect: diff={diff}"
```

### Fix 7: Seed Competition Test (MANDATORY - Rev 4 STRICTER)

**Committee-recommended test to validate the seed HELPS the architecture.**

> "If seeded is no better than random, you're just grafting a competent transformer onto expensive noise. The seed must be BETTER."

**Purpose**: Detect seed rejection BEFORE it happens in production.

**Rev 4 Change**: Seeded model must show **measurable advantage**, not just "not worse".

```python
def seed_competition_test(config, train_data, n_steps=100):
    """
    Validate that seeding HELPS rather than HINDERS training.
    BLOCKS TRAINING if seed doesn't provide advantage.

    Rev 4: Must be BETTER than random, not just "not worse".
    """
    # Model A: Seeded with SmolLM
    seeded_model = AtlasModel(config)
    seeded_model = apply_seed_initialization(seeded_model, config)

    # Model B: Random initialization (no seed)
    random_config = config.copy()
    random_config['seed_model']['enabled'] = False
    random_model = AtlasModel(random_config)

    # Train both for n_steps
    seeded_losses = train_n_steps(seeded_model, train_data, n_steps)
    random_losses = train_n_steps(random_model, train_data, n_steps)

    # Compare metrics
    seeded_final = np.mean(seeded_losses[-10:])
    random_final = np.mean(random_losses[-10:])
    seeded_volatility = np.std(seeded_losses)
    random_volatility = np.std(random_losses)
    seed_advantage = random_final - seeded_final  # Positive = seed helps

    logger.info(f"Seed Competition Test Results (n={n_steps} steps):")
    logger.info(f"  Seeded model:  final_loss={seeded_final:.4f}, volatility={seeded_volatility:.4f}")
    logger.info(f"  Random model:  final_loss={random_final:.4f}, volatility={random_volatility:.4f}")
    logger.info(f"  Seed advantage: {seed_advantage:+.4f} (positive = seed helps)")

    # Rev 4: Seeded MUST be better, not just "not worse"
    # If seeded is worse OR equal, something is wrong
    if seeded_final > random_final * 1.1:  # Seeded 10%+ worse = FAIL
        logger.error(
            f"SEED COMPETITION FAILED: Seeded model loss ({seeded_final:.4f}) is worse than "
            f"random ({random_final:.4f}). Architecture is rejecting the seed."
        )
        raise RuntimeError(
            "Seed competition test failed. The SmolLM seed is being rejected. "
            "Fix the memory initialization before proceeding."
        )

    if seed_advantage < 0.05:  # Seeded must be at least 0.05 loss better = NEW
        logger.error(
            f"SEED COMPETITION FAILED: Seeded model ({seeded_final:.4f}) shows no advantage "
            f"over random ({random_final:.4f}). Advantage: {seed_advantage:+.4f}."
        )
        raise RuntimeError(
            "Seed competition test failed. Seed provides no advantage - "
            "memory system may be bypassed. Check gate liveness first."
        )

    if seeded_volatility > random_volatility * 1.5:  # Seeded 50%+ more volatile
        logger.warning(
            f"SEED COMPETITION WARNING: Seeded model more volatile ({seeded_volatility:.4f}) "
            f"than random ({random_volatility:.4f}). Possible seed conflict."
        )

    logger.info(f"Seed Competition Test: PASSED - Seeded model shows {seed_advantage:.4f} advantage")
    return {
        'seeded_final': seeded_final,
        'random_final': random_final,
        'seeded_volatility': seeded_volatility,
        'random_volatility': random_volatility,
        'seed_advantage': seed_advantage
    }
```

**When to Run**: Pre-flight check before ANY seeded training run.

**New Threshold (Rev 4)**: `seed_advantage >= 0.05` (seeded must be at least 0.05 loss better)

**What This Proves**:
- If seeded model trains **faster/smoother** → Seed is helping, proceed
- If seeded model trains **same as random** → Memory bypassed, DO NOT PROCEED (Rev 4 NEW)
- If seeded model trains **slower/rougher** → Architecture is fighting seed, DO NOT PROCEED

**The Organ Transplant Analogy**: This is the immunosuppressant test. The organ isn't just present - it must be functioning.

### Fix 8: Gradient Health Check on M_init (MANDATORY - Rev 4 NEW)

**Committee concern**: "Dead gates tells you information can't flow THROUGH the memory. But is the memory itself receiving gradients? You need to prove the organ is alive, not just present."

**Purpose**: Verify M_init matrices are receiving gradients during training.

```python
def check_m_init_gradient_health(model, threshold=1e-6):
    """
    Verify memory initialization matrices are receiving gradients.
    BLOCKS TRAINING if gradients are dead.
    """
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'm_init'):
            if block.m_init.grad is None:
                logger.error(f"M_init gradient is None in layer {i}")
                return False

            grad_norm = block.m_init.grad.norm().item()
            if grad_norm < threshold:
                logger.error(
                    f"M_init gradient dead in layer {i}: "
                    f"grad_norm={grad_norm:.2e} < {threshold:.2e}"
                )
                return False

            logger.info(f"Layer {i} M_init grad_norm: {grad_norm:.4f}")

    return True

# In training loop - run after first backward pass
if step == 1:
    if not check_m_init_gradient_health(model):
        logger.error("KILL SWITCH: M_init is not receiving gradients")
        sys.exit(1)
```

**When to Run**: After first backward pass (step 1), then every 1000 steps

**Why This Matters**:
- Dead gates prevent information from flowing THROUGH memory (Fix 1)
- But even with live gates, M_init might not receive gradients
- If M_init grad_norm ≈ 0, the seed weights are frozen - transplant failed

### Fix 9: M3 Alpha Orthogonality Check (MANDATORY - Rev 4 NEW)

**Committee concern**: "If alpha starts at 0.5 and the global/local memories collapse to the same frequency, alpha becomes arbitrary. Need orthogonality check."

**Purpose**: Verify global and local memory gradients are orthogonal (learning different things).

```python
def check_m3_orthogonality(model, threshold=0.1):
    """
    Check that global and local memory are learning different things.
    If gradients are parallel, alpha is meaningless.
    """
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'm3_mixer'):
            global_grad = block.global_memory.weight.grad
            local_grad = block.local_memory.weight.grad

            if global_grad is None or local_grad is None:
                continue

            # Compute cosine similarity between gradient directions
            cos_sim = F.cosine_similarity(
                global_grad.flatten().unsqueeze(0),
                local_grad.flatten().unsqueeze(0)
            ).item()

            logger.info(f"Layer {i} global/local gradient cos_sim: {cos_sim:.4f}")

            if abs(cos_sim) > 0.9:  # Gradients nearly parallel
                logger.warning(
                    f"Layer {i} frequency collapse risk: cos_sim={cos_sim:.4f}. "
                    "Global and local memory learning same thing."
                )
                return False, cos_sim

    return True, None
```

**When to Run**: Every 1000 steps during training

**Threshold**: cos_sim < 0.9 (gradients must be somewhat orthogonal)

### Fix 10: Alpha Guardrails Kill Switch (MANDATORY - Rev 4 NEW)

**Committee concern**: "If alpha pins to 0.99 or 0.01, you've collapsed to single-memory mode. That's failure."

**Purpose**: Kill training if alpha collapses to extremes.

```python
class AlphaGuardrails:
    """
    Monitor M3 mixing alpha values.
    Kill if alpha pins to extremes (frequency collapse).
    """
    def __init__(self, lower=0.01, upper=0.99, patience=500):
        self.lower = lower
        self.upper = upper
        self.patience = patience
        self.steps_at_extreme = 0

    def check(self, model, step):
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'm3_mixer'):
                alpha = block.m3_mixer.alpha.item()

                if alpha < self.lower or alpha > self.upper:
                    self.steps_at_extreme += 1
                    logger.warning(
                        f"Step {step}: Layer {i} alpha={alpha:.4f} at extreme "
                        f"({self.steps_at_extreme}/{self.patience} steps)"
                    )

                    if self.steps_at_extreme >= self.patience:
                        logger.error(
                            f"KILL SWITCH: Alpha pinned at {alpha:.4f} for "
                            f"{self.patience} steps. Frequency collapse detected."
                        )
                        save_diagnostic_checkpoint(model, step, reason="alpha_collapse")
                        sys.exit(1)
                else:
                    self.steps_at_extreme = 0  # Reset counter
```

**Thresholds**: 0.01 < alpha < 0.99 (must maintain healthy mixing)

**Patience**: 500 steps (brief excursions OK, sustained pinning = kill)

### Fix 11: Boundary Stress Test (MANDATORY - Rev 4 NEW)

**Committee concern**: "TNT resets combined with high gain at shard boundaries could cause gradient explosion. Stress test this BEFORE full training."

**Purpose**: Verify model survives shard boundaries with new gain=2.0.

```python
def boundary_stress_test(model, train_data, n_boundaries=20):
    """
    Stress test shard boundary handling with high gain.
    Verify no gradient explosions or NaN.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    memory_states = [None] * n_layers
    max_grad_norm = 0.0
    nan_detected = False

    for boundary_idx in range(n_boundaries):
        # Get chunk at shard boundary
        chunk = get_chunk_at_boundary(train_data, boundary_idx)

        optimizer.zero_grad()
        logits, memory_states, _ = model(
            chunk['input_ids'],
            memory_states=memory_states,
            shard_boundary=True  # Force boundary reset
        )

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), chunk['labels'].view(-1))

        if torch.isnan(loss):
            logger.error(f"NaN detected at boundary {boundary_idx}")
            nan_detected = True
            break

        loss.backward()

        # Track gradient magnitude
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        max_grad_norm = max(max_grad_norm, total_grad_norm)

        logger.info(f"Boundary {boundary_idx}: loss={loss.item():.4f}, grad_norm={total_grad_norm:.4f}")

        # Detach memory states
        memory_states = [s.detach() if s is not None else None for s in memory_states]

    if nan_detected:
        raise RuntimeError("Boundary stress test FAILED: NaN detected")

    if max_grad_norm > 100.0:  # Gradient explosion threshold
        raise RuntimeError(
            f"Boundary stress test FAILED: Max grad_norm={max_grad_norm:.2f} > 100.0. "
            "Gradient explosion at shard boundaries."
        )

    logger.info(f"Boundary stress test PASSED: max_grad_norm={max_grad_norm:.4f}")
    return max_grad_norm
```

**When to Run**: Pre-flight check before training

**Threshold**: max_grad_norm < 100.0 (no gradient explosion)

### Fix 12: Pre-Step-1 Memory Ablation (MANDATORY - Rev 4 NEW)

**Committee concern**: "Run memory ablation BEFORE step 1 to establish continuity. Then at step 1000, if the number changed, you know training affected memory."

**Purpose**: Establish memory impact baseline before any training.

```python
def pre_training_memory_check(model, val_data):
    """
    Establish memory impact baseline BEFORE training starts.
    This is the "pre-operative" measurement.
    """
    ppl_with = evaluate(model, val_data, use_memory=True)
    ppl_without = evaluate(model, val_data, use_memory=False)
    impact = ppl_without - ppl_with

    logger.info("PRE-TRAINING Memory Ablation:")
    logger.info(f"  With memory:    PPL = {ppl_with:.2f}")
    logger.info(f"  Without memory: PPL = {ppl_without:.2f}")
    logger.info(f"  Initial impact: {impact:+.2f} PPL")

    # Store baseline for later comparison
    return {
        'ppl_with': ppl_with,
        'ppl_without': ppl_without,
        'impact': impact,
        'timestamp': 'pre_training'
    }

# In training script:
baseline = pre_training_memory_check(model, val_data)

# At step 1000, compare:
step_1000 = validate_memory_impact(model, val_data, step=1000)
delta = step_1000['impact'] - baseline['impact']
logger.info(f"Memory impact change after 1000 steps: {delta:+.2f} PPL")

if delta < -baseline['impact'] * 0.5:  # Impact dropped by 50%+
    logger.error("Memory impact dropping - possible rejection in progress")
```

**When to Run**: Immediately before step 1 (before any optimizer.step())

**What This Proves**: Establishes the "before" measurement so we can detect regression.

---

## Pre-Training Checklist for Run #2 (Rev 4 - Second Committee Feedback)

### Pre-Flight Gates (ALL MUST PASS - NO EXCEPTIONS)

These are **go/no-go gates**, not diagnostics. If any fails, DO NOT LAUNCH.

| Gate | Name | Threshold | Action if Failed |
|------|------|-----------|------------------|
| **Gate 1** | PoL (Proof of Life) | All gates produce \|tanh(x)\| > 0.9 | Don't launch |
| **Gate 2** | Baton Pass | First-token penalty ≤ **2%** (Rev 4) | Don't launch |
| **Gate 3** | Seed Competition | Seeded shows **≥0.05 advantage** (Rev 4) | Don't launch |
| **Gate 4** | Memory Ablation | Memory removal causes ≥10 PPL spike | Don't launch |
| **Gate 5** | Boundary Stress | max_grad_norm < 100 over 20 boundaries (Rev 4) | Don't launch |
| **Gate 6** | Pre-Step Continuity | Memory impact baseline established (Rev 4) | Don't launch |

### Gate Dependencies (Rev 4 - Updated)

```
Gate 1 (PoL) ──┬── PASS ──→ Gate 3 (Seed Competition)
               │            │
               │            └── Seeded must show ≥0.05 advantage
               │
               └── FAIL ──→ SKIP Gate 3 (pointless if gates are dead)

Gate 2 (Baton Pass) ──→ Independent (tests M3 mixing, threshold=2%)

Gate 4 (Memory Ablation) ──→ Independent (tests memory effect)

Gate 5 (Boundary Stress) ──→ Independent (tests gain=2.0 + TNT reset)

Gate 6 (Pre-Step Continuity) ──→ Last gate, establishes baseline
```

**Why Gate 3 depends on Gate 1**: Seed Competition tests whether SmolLM initialization helps. But if gates are dead, information can't flow through the memory system, so the seed is irrelevant. We proved this: with dead gates, seeded and random models converge to identical loss.

**Rev 4 Changes**:
- Gate 2 threshold tightened from 5% → **2%** (committee: "5% is still too high")
- Gate 3 now requires **measurable advantage** (seed_advantage ≥ 0.05), not just "not worse"
- Gate 5 added: Boundary stress test with gain=2.0 (committee: "stress test TNT boundaries")
- Gate 6 added: Pre-step continuity check (committee: "baseline before step 1")

**Checkpoint Testing**: At training checkpoints, run Gate 1 (PoL) first. If it fails, the model is broken - no need to run Seed Competition.

```bash
# Pre-flight script (MANDATORY before any Run #2 attempt)
python scripts/preflight_gates.py --config configs/default.yaml
# Must see: "ALL GATES PASSED. CLEAR FOR LAUNCH."
# If ANY gate fails: DO NOT PROCEED.
```

### Implementation Checklist (Rev 4 - Expanded)

Before starting training:

**Core Fixes (Blocking):**
- [ ] Input gain initialization updated to 2.0 (Fix 2) ⚠️ SEE RISK NOTE
- [ ] Gate liveness check passes at initialization (Fix 1)
- [ ] Isolation suite includes gate saturation test (Fix 6)

**Kill Switches:**
- [ ] Activation kill switch installed in training loop (Fix 3)
- [ ] Memory ablation kill switch installed (Fix 4)
- [ ] Alpha guardrails kill switch installed (Fix 10 - NEW)

**Pre-Flight Gates:**
- [ ] Baton pass gate passes pre-flight @ **2% threshold** (Fix 5 - TIGHTENED)
- [ ] Seed competition shows **≥0.05 advantage** (Fix 7 - STRICTER)
- [ ] Boundary stress test passes (Fix 11 - NEW)
- [ ] Pre-step-1 memory ablation baseline captured (Fix 12 - NEW)

**Training Loop Monitors:**
- [ ] M_init gradient health check at step 1 (Fix 8 - NEW)
- [ ] M3 orthogonality check every 1000 steps (Fix 9 - NEW)

### Kill Switches Active During Training (Rev 4 - Expanded)

These run automatically and will **terminate the run** if triggered:

| Kill Switch | Trigger | Patience | Effect |
|-------------|---------|----------|--------|
| Activation | Mean \|x\| < 0.1 any layer | 1000 steps | `sys.exit(1)` |
| Memory Ablation | Impact < 10 PPL at checkpoints | Immediate | `sys.exit(1)` |
| M_init Gradient | grad_norm < 1e-6 at step 1 | Immediate | `sys.exit(1)` (NEW) |
| Alpha Guardrails | alpha < 0.01 or > 0.99 | 500 steps | `sys.exit(1)` (NEW) |
| M3 Orthogonality | cos_sim > 0.9 (gradient collapse) | Warning only | Log alert (NEW) |

> "A crash is infinitely better than a coma."

---

## Metrics Dashboard Updates

Add to Streamlit dashboard:

1. **Gate Health Panel**
   - Tanh input magnitude per layer (with 0.1 threshold line)
   - % of activations > 0.5

2. **Memory Effectiveness Panel**
   - Memory vs no-memory PPL comparison
   - First-token loss vs rest-of-chunk loss

3. **Alert System**
   - Red alert if gates go linear
   - Yellow alert if memory impact drops

---

## Lessons Learned (Rev 2 - Expanded)

### From Run #1 Analysis

1. **"No problems" can BE the problem**: Zero saturation sounded healthy but indicated dead gates

2. **Aggregate loss hides component failures**: Loss went down, but memory wasn't contributing

3. **Validation must match training**: Eval loop must use identical memory reset schedule

4. **Liveness checks are mandatory**: Not just "is it stable" but "is it working"

5. **Committee review is valuable**: External analysis caught issues we normalized

### From Committee Feedback (Rev 2)

6. **Monitoring ≠ Enforcement**: Passive alerts get ignored at 3am. Kill switches don't.

7. **Unit tests don't catch emergent behaviors**: All tests passed, but the system failed. Need system-level invariants.

8. **Seed initialization is an organ transplant**: The body (optimizer) can reject the organ (new architecture). Must run compatibility tests first.

9. **"Acceptable" thresholds must be questioned**: We called 11% first-token penalty "acceptable". It wasn't. It meant memory wasn't working.

10. **Crashes > Comas**: A run that terminates with an error message is infinitely more useful than one that silently wastes compute.

### The Ferrari/Lawnmower Principle

> "If your nonlinear is behaving linearly, you don't have a gate. You have expensive arithmetic. It's like buying a Ferrari engine to power a lawnmower."

The O(d²) polynomial expansion, QK projection, and memory architecture only matter if they're actually being used. Compute cost without functionality is pure waste.

---

## Committee Feedback - Action Item Summary

### Rev 2 Actions (First Committee Feedback)

| Feedback | Original Fix | Rev 2 Fix |
|----------|--------------|-----------|
| "Stop monitoring, start enforcing" | Fix 3: Log warning | Fix 3: `sys.exit(1)` after 1000 steps |
| "Automate the ablation consequence" | Fix 4: Log warning | Fix 4: `sys.exit(1)` if <10 PPL impact |
| "Validate seed doesn't cause conflict" | Not addressed | Fix 7: Seed competition test |
| "11% baton pass is terrifying" | Fix 5: Log metric | Fix 5: Hard gate ≤5%, blocks launch |
| "Make failure loud" | Dashboard alerts | All checks become kill switches |

### Rev 4 Actions (Second Committee Feedback)

| Feedback | Rev 2 State | Rev 4 Fix |
|----------|-------------|-----------|
| "Gain=2.0 might cause FASTER rejection" | Fix 2: gain=2.0 | Fix 2: Added risk note + graduated schedule option |
| "Seed must be BETTER, not just 'not worse'" | Fix 7: ≤ random | Fix 7: Seeded must show ≥0.05 advantage |
| "Need gradient health check on M_init" | Not addressed | Fix 8: M_init grad_norm check at step 1 |
| "5% baton pass still too high" | Fix 5: ≤5% | Fix 5: **≤2%** or recovery slope |
| "Alpha at 0.5 risks frequency collapse" | Not addressed | Fix 9: Orthogonality check, Fix 10: Alpha guardrails |
| "TNT + high gain = gradient explosion risk" | Not addressed | Fix 11: Boundary stress test |
| "Need pre-step-1 baseline" | Not addressed | Fix 12: Pre-training memory ablation baseline |

---

## Timeline

1. **Now → 50K steps**: Continue Run #1 for data collection only
2. **After 50K**: Implement Fixes 1-7 per this PRD
3. **Before Run #2**:
   - All pre-flight gates must pass
   - `scripts/preflight_gates.py` must report "CLEAR FOR LAUNCH"
4. **Run #2**: With kill switches active, no zombies allowed

---

## Submission Request

The committee requested:

> "We would love to see the results from that component isolation test suite, especially the baton pass metrics, before you launch the full run. Submit those logs, and we can take another look."

**Deliverable for next review (Rev 4 - Updated)**: Pre-flight gate results showing:
- [ ] Gate 1: Liveness test output (ALL layers must reach |tanh| > 0.9)
- [ ] Gate 2: Baton pass penalty measurement (target: **≤2%**, not 5%)
- [ ] Gate 3: Seed competition test results (target: **≥0.05 advantage**, not just "not worse")
- [ ] Gate 4: Memory ablation test results (target: ≥10 PPL impact)
- [ ] Gate 5: Boundary stress test (target: max_grad_norm < 100) (NEW)
- [ ] Gate 6: Pre-step-1 memory ablation baseline captured (NEW)
- [ ] M_init gradient health check after step 1 (NEW)

**Additional validation (Rev 4)**:
- [ ] If gain=2.0 causes issues in stress test, present graduated gain schedule as alternative
- [ ] Show M3 orthogonality check results (cos_sim < 0.9)
- [ ] Demonstrate alpha stays within [0.01, 0.99] range

---

## Pre-Flight Baseline Results (Rev 3 - 2026-01-21)

Ran `scripts/preflight_gates.py` on current architecture (fresh model, no training):

```
============================================================
PRE-FLIGHT GATE SUMMARY
============================================================

  [FAIL]  Gate Liveness
          Layers [0, 1, 2, 3, 5] cannot reach nonlinear operation.
          Gates are structurally dead.

  [PASS]  Baton Pass
          Penalty 0.2% is within 5.0% threshold

  [WARN]  Seed Competition
          SKIPPED (run separately)

  [PASS]  Memory Ablation
          Memory contributes 383.5 PPL (threshold: 10.0)

============================================================
GATE CHECK FAILED. DO NOT LAUNCH.
============================================================
```

### Detailed Gate 1 Results (Gate Liveness)

```
Layer 0: max|output| = 0.7707 -> DEAD
Layer 1: max|output| = 0.7766 -> DEAD
Layer 2: max|output| = 0.7913 -> DEAD
Layer 3: max|output| = 0.7862 -> DEAD
Layer 4: max|output| = 0.9044 -> ALIVE
Layer 5: max|output| = 0.8161 -> DEAD
```

**Analysis**: Even with 10x input magnitude, 5 of 6 layers cannot produce |output| > 0.9. The gates are **structurally incapable** of nonlinear operation with current initialization.

### What This Tells Us

| Gate | Result | Implication |
|------|--------|-------------|
| Gate Liveness | **FAILED** | `input_gain=0.5` is too low; gates can't saturate |
| Baton Pass | PASSED | M3 mixing works structurally (0.2% penalty) |
| Memory Ablation | PASSED | Memory affects output in fresh model (+383 PPL) |

**Key Insight**: The architecture is sound EXCEPT for the gate initialization. Fix 2 (`input_gain` → 2.0) should resolve Gate Liveness, which will allow the memory system to function.

### Seed Competition Results (100 steps each)

```
Metric                    SEEDED          RANDOM          Winner
----------------------------------------------------------------------
Initial loss              11.3301         11.3614         SEEDED
Final loss (last 10)      8.1761          8.1308          RANDOM
Loss reduction            3.1540          3.2306          RANDOM
Volatility (std)          0.8132          0.7909          RANDOM
Seed advantage            -0.0453         (negative = seed HURT)
```

**Result (Rev 2 criteria)**: PASSED (technically) - seeded model is not significantly worse than random.

**⚠️ Result (Rev 4 criteria)**: **WOULD FAIL** - seed_advantage = -0.0453, required ≥ 0.05

**But critically**: The seed provides NO advantage. Both models converge to nearly identical loss.

**Why This Matters**: This confirms the dead gates hypothesis. With non-functional gates:
- The memory system is bypassed
- M_init (seeded from SmolLM) is never used
- Seed vs random initialization is irrelevant
- Under Rev 4 criteria, this is now a **FAILED** gate

**After Fix 1 (input_gain → 2.0)**: Re-run this test. With functional gates, the seeded model SHOULD outperform random because it starts with meaningful memory matrices. If it still doesn't show ≥0.05 advantage, there's a deeper problem with how the seed is integrated.

---

## Code Changes Required (Consolidated)

### Change 1: Input Gain Initialization (BLOCKING)

**File**: `src/qk_projection.py`

**Line ~59** (in `QKProjectionLayer.__init__`):

```python
# CURRENT (broken - gates cannot saturate)
self.input_gain = nn.Parameter(torch.ones(d_model) * 0.5)

# REQUIRED (enables nonlinear operation)
self.input_gain = nn.Parameter(torch.ones(d_model) * 2.0)
```

**Why**: With `input_gain=0.5`, the maximum tanh output is ~0.77 even with 10x input. With `input_gain=2.0`, gates can reach full saturation range.

**Verification**: Re-run `preflight_gates.py` - Gate Liveness must PASS.

---

### Change 2: Add Activation Magnitude Tracking (REQUIRED)

**File**: `src/qk_projection.py`

**Add to `QKProjectionLayer`**:

```python
# In __init__:
self.register_buffer('last_input_magnitude', torch.tensor(0.0))

# In forward(), after computing q_gated:
self.last_input_magnitude = q_gated.abs().mean().detach()
```

**Why**: Needed for the activation kill switch to monitor gate health during training.

---

### Change 3: Add Activation Kill Switch to Training Loop (REQUIRED)

**File**: `scripts/train.py`

**Add class** (see Fix 3 in PRD for full implementation):

```python
class ActivationKillSwitch:
    def __init__(self, threshold=0.1, patience=1000):
        ...
    def check(self, model, step):
        # Check activation magnitudes
        # sys.exit(1) if below threshold for patience steps
```

**Add to training loop**:

```python
kill_switch = ActivationKillSwitch(threshold=0.1, patience=1000)

for step in range(total_steps):
    # ... training code ...

    if step % 100 == 0:
        kill_switch.check(model, step)
```

---

### Change 4: Add Memory Ablation Kill Switch (REQUIRED)

**File**: `scripts/train.py`

**Add function** (see Fix 4 in PRD for full implementation):

```python
def validate_memory_impact(model, val_data, step, min_impact=10.0):
    # Compare PPL with/without memory
    # sys.exit(1) if impact < min_impact
```

**Add to training loop** at steps 1K, 5K, 10K, 25K, 50K:

```python
if step in [1000, 5000, 10000, 25000, 50000]:
    validate_memory_impact(model, val_data, step)
```

---

### Change 5: Pre-Flight Script Already Created

**File**: `scripts/preflight_gates.py` (DONE)

**Usage**:
```bash
python scripts/preflight_gates.py --config configs/default.yaml
# Must see: "ALL GATES PASSED. CLEAR FOR LAUNCH."
```

---

## Implementation Order (Rev 4 - Updated)

**Phase 1: Core Architecture Fixes**
1. **Change 1** (input_gain → 2.0) - FIRST, enables all other fixes to matter
2. **Change 2** (magnitude tracking) - Enables kill switch
3. Re-run pre-flight gates - Verify Gate Liveness PASSES

**Phase 2: Kill Switches**
4. **Change 3** (activation kill switch) - Prevents silent failure
5. **Change 4** (memory ablation kill switch) - Prevents zombie runs
6. **Change 6** (alpha guardrails) - Prevents frequency collapse (NEW)

**Phase 3: Gradient Health (NEW)**
7. **Change 5** (M_init gradient check) - Verify organ is alive
8. **Change 7** (M3 orthogonality check) - Monitor frequency collapse

**Phase 4: Pre-Flight Validation**
9. Run boundary stress test (Fix 11) - Verify gain=2.0 doesn't explode
10. Run seed competition test - Verify seed shows ≥0.05 advantage
11. Run pre-step-1 memory ablation - Establish baseline
12. Full pre-flight pass required before Run #2

---

## New Code Changes (Rev 4)

### Change 5: M_init Gradient Health Check (NEW)

**File**: `scripts/train.py`

**Add after first backward pass**:
```python
def check_m_init_gradient_health(model, threshold=1e-6):
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'm_init'):
            if block.m_init.grad is None or block.m_init.grad.norm() < threshold:
                return False
    return True

# In training loop at step 1:
if step == 1:
    if not check_m_init_gradient_health(model):
        logger.error("KILL SWITCH: M_init is not receiving gradients")
        sys.exit(1)
```

---

### Change 6: Alpha Guardrails (NEW)

**File**: `scripts/train.py`

**Add class** (see Fix 10 for full implementation):
```python
class AlphaGuardrails:
    def __init__(self, lower=0.01, upper=0.99, patience=500):
        ...
    def check(self, model, step):
        # Check alpha values
        # sys.exit(1) if pinned at extreme for patience steps
```

**Add to training loop**:
```python
alpha_guardrails = AlphaGuardrails(lower=0.01, upper=0.99, patience=500)

for step in range(total_steps):
    # ... training code ...
    if step % 100 == 0:
        alpha_guardrails.check(model, step)
```

---

### Change 7: M3 Orthogonality Check (NEW)

**File**: `scripts/train.py`

**Add function** (see Fix 9 for full implementation):
```python
def check_m3_orthogonality(model, threshold=0.9):
    # Check cosine similarity between global/local gradients
    # Return False if gradients are too parallel
```

**Add to training loop** every 1000 steps:
```python
if step % 1000 == 0:
    orthogonal, cos_sim = check_m3_orthogonality(model)
    if not orthogonal:
        logger.warning(f"M3 frequency collapse risk: cos_sim={cos_sim}")
```

---

### Change 8: Pre-Flight Script Updates (NEW)

**File**: `scripts/preflight_gates.py`

**Add new gates**:
1. Gate 5: Boundary stress test (20 boundaries, check grad_norm < 100)
2. Gate 6: Pre-step continuity check (establish memory ablation baseline)
3. Update Gate 2 threshold from 5% → 2%
4. Update Gate 3 to require seed_advantage ≥ 0.05

---

*Document created: 2026-01-21*
*Rev 2 (Committee Feedback): 2026-01-21*
*Rev 3 (Pre-Flight Baseline): 2026-01-21*
*Rev 4 (Second Committee Feedback): 2026-01-21*
