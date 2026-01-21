#!/usr/bin/env python3
"""
Pre-Flight Gates for Atlas Run #2 (Rev 4)

These are GO/NO-GO gates, not diagnostics.
If ANY gate fails, DO NOT LAUNCH the training run.

Gates:
1. PoL (Proof of Life) - Can tanh gates actually reach nonlinear territory?
2. Baton Pass - Does M3 transfer context across shard boundaries? (threshold: 2%)
3. Seed Competition - Is the seed helping or fighting the architecture? (need ≥0.05 advantage)
4. Memory Ablation - Does memory actually affect model output?

Run from atlas_tnt directory:
    python scripts/preflight_gates.py --config configs/default.yaml

Expected output:
    ALL GATES PASSED. CLEAR FOR LAUNCH.

If any gate fails:
    DO NOT PROCEED. Debug the failure first.

Rev 4 Changes:
- Baton pass threshold tightened from 5% to 2%
- Seed competition requires ≥0.05 advantage (not just "not worse")
- Memory ablation distinguishes "no effect" (FAIL) from "negative effect" (WARNING)

Per committee feedback:
    "A crash is infinitely better than a coma."
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import sys
import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Setup imports
project_root = Path(__file__).parent.parent.resolve()
os.chdir(project_root)
sys.path.insert(0, str(project_root))


class GateResult(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"


@dataclass
class GateReport:
    name: str
    result: GateResult
    message: str
    details: Dict


def load_model_and_data(config_path: str, device: torch.device):
    """Load model and validation data for testing."""
    import runpy

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load model class
    train_globals = runpy.run_path('scripts/train.py', run_name='__train__')
    AtlasModel = train_globals['AtlasModel']

    # Build fresh model (no checkpoint - we're testing architecture, not trained weights)
    model = AtlasModel(config).to(device)
    model.requires_grad_(False)

    # Load validation data
    val_tokens = np.load('data/val.npy')

    return model, config, val_tokens


# ============================================================================
# GATE 1: Gate Liveness
# ============================================================================

def gate_liveness_test(model, device) -> GateReport:
    """
    Gate 1: PoL (Proof of Life) - Verify tanh gates can reach nonlinear territory.

    The tanh function is nonlinear at the extremes but linear near zero:
        tanh(0.1) = 0.1   (linear region - gate is DEAD)
        tanh(1.0) = 0.76  (nonlinear region - gate is ALIVE)
        tanh(2.0) = 0.96  (saturation region - gate is ACTIVE)

    Test: Inject high-magnitude input and verify gates can produce |output| > 0.9

    PASS: All layers can saturate
    FAIL: Any layer cannot reach nonlinear operation

    NOTE: Gate 3 (Seed Competition) depends on this gate passing.
    If PoL fails, Seed Competition is skipped (pointless with dead gates).
    """
    print("\n" + "=" * 60)
    print("GATE 1: PoL (PROOF OF LIFE)")
    print("=" * 60)
    print("Testing if tanh gates can reach nonlinear territory...")
    print()

    d_model = model.blocks[0].d_model
    n_layers = len(model.blocks)

    # Generate high-magnitude test input
    # We want to verify the gates CAN saturate, not that they DO in normal operation
    test_magnitudes = [1.0, 2.0, 5.0, 10.0]

    results = {}
    all_passed = True

    for layer_idx, block in enumerate(model.blocks):
        layer_results = {}

        for mag in test_magnitudes:
            # Create test input
            x = torch.randn(1, 32, d_model, device=device) * mag

            # Get the QK projection layer
            qk_layer = block.qk_proj_layer

            # Create dummy P matrix and k for the projection
            P = torch.eye(d_model, device=device).unsqueeze(0) * 0.01
            k = torch.randn(1, 32, d_model, device=device)
            q = x  # Use scaled input as query

            with torch.no_grad():
                # Run through QK projection
                q_aligned, _, _ = qk_layer(q, k, P)

                # The output goes through tanh internally
                # Check the max absolute value
                max_val = q_aligned.abs().max().item()
                layer_results[mag] = max_val

        # Check if this layer can reach saturation (|output| > 0.9) with any input magnitude
        can_saturate = any(v > 0.9 for v in layer_results.values())
        max_achieved = max(layer_results.values())

        status = "ALIVE" if can_saturate else "DEAD"
        if not can_saturate:
            all_passed = False

        print(f"  Layer {layer_idx}: max|output| = {max_achieved:.4f} -> {status}")
        results[layer_idx] = {
            'max_output': max_achieved,
            'can_saturate': can_saturate,
            'by_magnitude': layer_results
        }

    print()

    if all_passed:
        return GateReport(
            name="PoL (Proof of Life)",
            result=GateResult.PASSED,
            message="All layers can reach nonlinear operation",
            details=results
        )
    else:
        dead_layers = [i for i in range(n_layers) if not results[i]['can_saturate']]
        return GateReport(
            name="PoL (Proof of Life)",
            result=GateResult.FAILED,
            message=f"Layers {dead_layers} cannot reach nonlinear operation. Gates are structurally dead.",
            details=results
        )


# ============================================================================
# GATE 2: Baton Pass
# ============================================================================

def baton_pass_gate(model, val_tokens, config, device, n_shards: int = 10) -> GateReport:
    """
    Gate 2: Verify M3 mixing transfers context across shard boundaries.

    At a shard boundary, memory is reset/mixed. The first token after the boundary
    has zero local context and must rely entirely on the passed state.

    If first-token loss >> rest-of-chunk loss, the baton pass is failing.

    THRESHOLD: First-token penalty must be <= 2% (Rev 4 - tightened from 5%)

    Per committee: "If you're dropping the baton every 2,048 tokens, you're not
    running a marathon. You're running a series of disconnected sprints."
    "5% is still too high. Drop to 2%."
    """
    print("\n" + "=" * 60)
    print("GATE 2: BATON PASS TEST")
    print("=" * 60)
    print("Testing M3 state transfer across shard boundaries...")
    print()

    MAX_PENALTY = 0.02  # 2% maximum acceptable penalty (Rev 4 - tightened from 5%)

    chunk_size = config['training']['stage1']['chunk_size']
    batch_size = config['training']['batch_size']
    shard_length = config['training']['stage1']['shard_length']
    n_layers = config['model']['n_layers']

    chunks_per_shard = shard_length // chunk_size

    first_token_losses = []
    rest_token_losses = []

    memory_states = [None] * n_layers

    with torch.no_grad():
        for shard_idx in range(n_shards):
            for chunk_in_shard in range(chunks_per_shard):
                batch_idx = shard_idx * chunks_per_shard + chunk_in_shard

                # Shard boundary at start of each shard (except first)
                shard_boundary = (chunk_in_shard == 0) and (shard_idx > 0)

                # Build batch
                input_ids_list = []
                labels_list = []
                for b in range(batch_size):
                    chunk_idx = batch_idx * batch_size + b
                    start = chunk_idx * chunk_size
                    end = start + chunk_size + 1
                    if end > len(val_tokens):
                        break
                    chunk = val_tokens[start:end]
                    input_ids_list.append(torch.tensor(chunk[:-1], dtype=torch.long))
                    labels_list.append(torch.tensor(chunk[1:], dtype=torch.long))

                if len(input_ids_list) < batch_size:
                    break

                input_ids = torch.stack(input_ids_list).to(device)
                labels = torch.stack(labels_list).to(device)

                # Forward pass
                logits, memory_states, _ = model(
                    input_ids, memory_states=memory_states, shard_boundary=shard_boundary
                )

                # Compute per-token losses
                log_probs = F.log_softmax(logits, dim=-1)
                token_losses = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

                # At shard boundary, analyze first token specially
                if shard_boundary:
                    first_token_loss = token_losses[:, 0].mean().item()
                    rest_token_loss = token_losses[:, 1:].mean().item()
                    first_token_losses.append(first_token_loss)
                    rest_token_losses.append(rest_token_loss)

    if not first_token_losses:
        return GateReport(
            name="Baton Pass",
            result=GateResult.FAILED,
            message="Could not collect baton pass data (not enough shards)",
            details={}
        )

    avg_first = np.mean(first_token_losses)
    avg_rest = np.mean(rest_token_losses)
    penalty = (avg_first / avg_rest) - 1.0

    print(f"  First-token loss (at boundary): {avg_first:.4f}")
    print(f"  Rest-of-chunk loss:             {avg_rest:.4f}")
    print(f"  Penalty: {penalty*100:+.1f}%")
    print(f"  Threshold: {MAX_PENALTY*100}%")
    print()

    details = {
        'first_token_loss': avg_first,
        'rest_token_loss': avg_rest,
        'penalty': penalty,
        'threshold': MAX_PENALTY,
        'n_boundaries_tested': len(first_token_losses)
    }

    if penalty <= MAX_PENALTY:
        return GateReport(
            name="Baton Pass",
            result=GateResult.PASSED,
            message=f"Penalty {penalty*100:.1f}% is within {MAX_PENALTY*100}% threshold",
            details=details
        )
    else:
        return GateReport(
            name="Baton Pass",
            result=GateResult.FAILED,
            message=f"Penalty {penalty*100:.1f}% exceeds {MAX_PENALTY*100}% threshold. M3 mixing is not transferring context.",
            details=details
        )


# ============================================================================
# GATE 3: Seed Competition
# ============================================================================

def seed_competition_gate(config, device, n_steps: int = 100) -> GateReport:
    """
    Gate 3: Verify the seed model helps rather than hinders training.

    Per committee: "Run the seeded Atlas model for 100 steps. Then run a randomly
    initialized Atlas model for 100 steps. Compare them. If the seeded model trains
    slower or shows way more volatility, it proves the architecture is fighting the seed."

    Rev 4: "If seeded is no better than random, you're just grafting a competent
    transformer onto expensive noise. The seed must be BETTER."

    PASS: Seeded model shows ≥0.05 loss advantage over random (Rev 4 - stricter)
    FAIL: Seeded model is worse OR shows no advantage
    """
    print("\n" + "=" * 60)
    print("GATE 3: SEED COMPETITION TEST")
    print("=" * 60)
    print(f"Comparing seeded vs random initialization for {n_steps} steps...")
    print()

    import runpy
    import copy

    # Load model class and data
    train_globals = runpy.run_path('scripts/train.py', run_name='__train__')
    AtlasModel = train_globals['AtlasModel']

    train_tokens = np.load('data/train.npy')

    chunk_size = config['training']['stage1']['chunk_size']
    batch_size = config['training']['batch_size']
    n_layers = config['model']['n_layers']

    def train_n_steps(model, n_steps, label):
        """Train for n steps and return loss history."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        losses = []
        memory_states = [None] * n_layers

        for step in range(n_steps):
            # Build batch
            input_ids_list = []
            labels_list = []
            for b in range(batch_size):
                chunk_idx = step * batch_size + b
                start = chunk_idx * chunk_size
                end = start + chunk_size + 1
                if end > len(train_tokens):
                    start = 0
                    end = chunk_size + 1
                chunk = train_tokens[start:end]
                input_ids_list.append(torch.tensor(chunk[:-1], dtype=torch.long))
                labels_list.append(torch.tensor(chunk[1:], dtype=torch.long))

            input_ids = torch.stack(input_ids_list).to(device)
            labels = torch.stack(labels_list).to(device)

            # Forward pass
            optimizer.zero_grad()
            logits, memory_states, _ = model(input_ids, memory_states=memory_states, shard_boundary=False)

            # Detach memory states
            memory_states = [s.detach() if s is not None else None for s in memory_states]

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if step % 20 == 0:
                print(f"    {label}: step {step}, loss = {loss.item():.4f}")

        return losses

    # Model A: Seeded (default config has seed enabled)
    print("  Training SEEDED model...")
    seeded_config = copy.deepcopy(config)
    seeded_config['seed_model']['enabled'] = True
    seeded_model = AtlasModel(seeded_config).to(device)
    seeded_losses = train_n_steps(seeded_model, n_steps, "seeded")
    del seeded_model
    torch.cuda.empty_cache()

    # Model B: Random (disable seed)
    print("\n  Training RANDOM model...")
    random_config = copy.deepcopy(config)
    random_config['seed_model']['enabled'] = False
    random_model = AtlasModel(random_config).to(device)
    random_losses = train_n_steps(random_model, n_steps, "random")
    del random_model
    torch.cuda.empty_cache()

    # Compare
    seeded_final = np.mean(seeded_losses[-10:])
    random_final = np.mean(random_losses[-10:])
    seeded_volatility = np.std(seeded_losses)
    random_volatility = np.std(random_losses)

    print()
    print(f"  Results after {n_steps} steps:")
    print(f"    Seeded: final_loss={seeded_final:.4f}, volatility={seeded_volatility:.4f}")
    print(f"    Random: final_loss={random_final:.4f}, volatility={random_volatility:.4f}")
    print()

    seed_advantage = random_final - seeded_final  # Positive = seed helps
    MIN_ADVANTAGE = 0.05  # Rev 4: Must show measurable advantage

    details = {
        'seeded_final_loss': seeded_final,
        'random_final_loss': random_final,
        'seeded_volatility': seeded_volatility,
        'random_volatility': random_volatility,
        'seed_advantage': seed_advantage,
        'min_advantage_required': MIN_ADVANTAGE,
        'n_steps': n_steps
    }

    print(f"    Seed advantage: {seed_advantage:+.4f} (need >={MIN_ADVANTAGE})")
    print()

    # Rev 4: Seeded must be BETTER, not just "not worse"
    # Fail if seeded is 10%+ worse
    if seeded_final > random_final * 1.1:
        return GateReport(
            name="Seed Competition",
            result=GateResult.FAILED,
            message=f"Seeded model ({seeded_final:.4f}) is worse than random ({random_final:.4f}). Architecture is rejecting the seed.",
            details=details
        )

    # Rev 4: Fail if seed shows no advantage
    if seed_advantage < MIN_ADVANTAGE:
        return GateReport(
            name="Seed Competition",
            result=GateResult.FAILED,
            message=f"Seeded model shows no advantage ({seed_advantage:+.4f} < {MIN_ADVANTAGE}). Memory may be bypassed. Check PoL.",
            details=details
        )

    # Warning if seeded is more volatile
    if seeded_volatility > random_volatility * 1.5:
        return GateReport(
            name="Seed Competition",
            result=GateResult.WARNING,
            message=f"Seeded shows advantage but is more volatile ({seeded_volatility:.4f} vs {random_volatility:.4f}). Possible instability.",
            details=details
        )

    return GateReport(
        name="Seed Competition",
        result=GateResult.PASSED,
        message=f"Seeded model shows {seed_advantage:.4f} advantage (loss {seeded_final:.4f} vs random {random_final:.4f})",
        details=details
    )


# ============================================================================
# GATE 4: Memory Ablation
# ============================================================================

def memory_ablation_gate(model, val_tokens, config, device, n_batches: int = 50) -> GateReport:
    """
    Gate 4: Verify memory actually affects model predictions.

    Test: Compare PPL with memory vs PPL without memory (fresh state each batch)

    Rev 4: Three possible outcomes:
    - PASS: Memory helps (impact >= 10 PPL)
    - WARNING: Memory hurts (impact < -10 PPL) - expected pre-training with functional gates
    - FAIL: Memory has no effect (|impact| < 10 PPL) - gates may be dead

    Per committee: "Run memory ablation BEFORE step 1 to establish continuity.
    Then at step 1000, if the number changed, you know training affected memory."
    """
    print("\n" + "=" * 60)
    print("GATE 4: MEMORY ABLATION TEST")
    print("=" * 60)
    print("Testing if memory actually affects predictions...")
    print()

    MIN_IMPACT = 10.0  # Minimum PPL impact (positive or negative) required

    chunk_size = config['training']['stage1']['chunk_size']
    batch_size = config['training']['batch_size']
    n_layers = config['model']['n_layers']

    def compute_ppl(reset_every_batch: bool, label: str):
        """Compute PPL with specified memory reset mode."""
        memory_states = [None] * n_layers
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx in range(n_batches):
                if reset_every_batch:
                    memory_states = [None] * n_layers

                input_ids_list = []
                labels_list = []
                for b in range(batch_size):
                    chunk_idx = batch_idx * batch_size + b
                    start = chunk_idx * chunk_size
                    end = start + chunk_size + 1
                    if end > len(val_tokens):
                        break
                    chunk = val_tokens[start:end]
                    input_ids_list.append(torch.tensor(chunk[:-1], dtype=torch.long))
                    labels_list.append(torch.tensor(chunk[1:], dtype=torch.long))

                if len(input_ids_list) < batch_size:
                    break

                input_ids = torch.stack(input_ids_list).to(device)
                labels = torch.stack(labels_list).to(device)

                logits, memory_states, _ = model(
                    input_ids, memory_states=memory_states, shard_boundary=reset_every_batch
                )

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                batch_tokens = input_ids.size(0) * input_ids.size(1)
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        ppl = np.exp(total_loss / total_tokens)
        print(f"    {label}: PPL = {ppl:.2f}")
        return ppl

    # Test with memory (accumulating state)
    ppl_with_memory = compute_ppl(reset_every_batch=False, label="With memory   ")

    # Test without memory (reset every batch)
    ppl_without_memory = compute_ppl(reset_every_batch=True, label="Without memory")

    impact = ppl_without_memory - ppl_with_memory

    print()
    print(f"  Memory impact: {impact:+.2f} PPL")
    print(f"  Threshold: >={MIN_IMPACT} PPL")
    print()

    details = {
        'ppl_with_memory': ppl_with_memory,
        'ppl_without_memory': ppl_without_memory,
        'impact': impact,
        'threshold': MIN_IMPACT,
        'n_batches': n_batches
    }

    # Rev 4: Three outcomes based on memory impact
    if impact >= MIN_IMPACT:
        # Memory helps - this is the ideal case
        return GateReport(
            name="Memory Ablation",
            result=GateResult.PASSED,
            message=f"Memory contributes {impact:.1f} PPL (threshold: {MIN_IMPACT})",
            details=details
        )
    elif impact <= -MIN_IMPACT:
        # Memory hurts - expected PRE-TRAINING with functional gates
        # The untrained memory is loud noise that disrupts the seed model
        # This is actually a GOOD sign - gates are working, memory affects output
        # Training should teach the memory to help instead of hurt
        return GateReport(
            name="Memory Ablation",
            result=GateResult.WARNING,
            message=f"Memory HURTS by {abs(impact):.1f} PPL. Expected pre-training with functional gates. Training should fix this.",
            details=details
        )
    else:
        # Memory has no effect - this means gates may be dead or bypassed
        return GateReport(
            name="Memory Ablation",
            result=GateResult.FAILED,
            message=f"Memory has no effect ({impact:+.1f} PPL, need |impact| >= {MIN_IMPACT}). Gates may be dead or bypassed.",
            details=details
        )


# ============================================================================
# MAIN: Run All Gates
# ============================================================================

def run_all_gates(config_path: str, skip_seed_test: bool = False):
    """
    Run all pre-flight gates.

    Returns True only if ALL gates pass.
    """
    print("=" * 60)
    print("ATLAS PRE-FLIGHT GATE CHECK")
    print("=" * 60)
    print()
    print("These are GO/NO-GO gates. If any fails, DO NOT LAUNCH.")
    print()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model and data
    print("Loading model and data...")
    model, config, val_tokens = load_model_and_data(config_path, device)

    reports: List[GateReport] = []

    # Gate 1: PoL (Proof of Life)
    pol_report = gate_liveness_test(model, device)
    reports.append(pol_report)

    # Gate 2: Baton Pass
    reports.append(baton_pass_gate(model, val_tokens, config, device))

    # Gate 3: Seed Competition
    # DEPENDENCY: Only run if Gate 1 (PoL) passed - otherwise pointless
    if skip_seed_test:
        print("\n[Skipping Seed Competition test (--skip-seed)]")
        reports.append(GateReport(
            name="Seed Competition",
            result=GateResult.WARNING,
            message="SKIPPED (use without --skip-seed to run)",
            details={}
        ))
    elif pol_report.result == GateResult.FAILED:
        print("\n[Skipping Seed Competition - Gate 1 (PoL) FAILED]")
        print("  Seed Competition is pointless if gates are dead.")
        print("  With dead gates, seed vs random converge to same loss.")
        reports.append(GateReport(
            name="Seed Competition",
            result=GateResult.WARNING,
            message="SKIPPED - Gate 1 (PoL) failed. Fix gates first.",
            details={'reason': 'dependency_failed', 'depends_on': 'Gate 1 (PoL)'}
        ))
    else:
        # PoL passed - Seed Competition is meaningful
        del model
        torch.cuda.empty_cache()
        reports.append(seed_competition_gate(config, device, n_steps=100))
        # Reload model for gate 4
        model, _, _ = load_model_and_data(config_path, device)

    # Gate 4: Memory Ablation
    reports.append(memory_ablation_gate(model, val_tokens, config, device))

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 60)
    print("PRE-FLIGHT GATE SUMMARY")
    print("=" * 60)
    print()

    all_passed = True
    has_warnings = False

    for report in reports:
        if report.result == GateResult.PASSED:
            status = "PASS"
        elif report.result == GateResult.WARNING:
            status = "WARN"
            has_warnings = True
        else:
            status = "FAIL"
            all_passed = False

        print(f"  [{status}]  {report.name}")
        print(f"          {report.message}")
        print()

    print("=" * 60)

    if all_passed and not has_warnings:
        print("ALL GATES PASSED. CLEAR FOR LAUNCH.")
        print("=" * 60)
        return True
    elif all_passed and has_warnings:
        print("GATES PASSED WITH WARNINGS. Review before launch.")
        print("=" * 60)
        return True
    else:
        print("GATE CHECK FAILED. DO NOT LAUNCH.")
        print()
        print("Per committee: 'A crash is infinitely better than a coma.'")
        print("Debug the failures before proceeding.")
        print("=" * 60)
        return False


def main():
    parser = argparse.ArgumentParser(description='Pre-flight gate check for Atlas Run #2')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--skip-seed', action='store_true',
                        help='Skip the seed competition test (faster but incomplete)')
    args = parser.parse_args()

    start_time = time.time()

    passed = run_all_gates(args.config, skip_seed_test=args.skip_seed)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
