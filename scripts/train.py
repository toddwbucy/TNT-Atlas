#!/usr/bin/env python3
"""
Atlas with TNT Training - Training Script

PRD v4.2 Implementation

Usage:
    # With streaming data
    python scripts/train.py --config configs/default.yaml

    # With pre-tokenized data (faster)
    python scripts/prepare_data.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --data-dir data/

    # Resume from checkpoint
    python scripts/train.py --config configs/default.yaml --resume runs/checkpoints/step_10000.pt

IMPORTANT: Run the isolation suite first!
    python scripts/run_isolation_suite.py
"""

import argparse
import json
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path for package imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas_block import AtlasMAGBlock
from src.cache_manager import HybridCacheManager
from src.m3_mixing import M3MixingState
from src.config_schema import load_config, AtlasConfig, ModelConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Kill Switches (PRD Rev 4 - Run #2 Fixes)
# ═══════════════════════════════════════════════════════════════════════════════

class ActivationKillSwitch:
    """
    PRD Fix 3: Monitor tanh gate activation magnitudes.
    TERMINATES if gates go dead (linear region).

    Per committee: "A crash is infinitely better than a coma."
    """

    def __init__(self, threshold: float = 0.1, patience: int = 1000):
        self.threshold = threshold
        self.patience = patience
        self.steps_below_threshold = 0

    def check(self, model: nn.Module, step: int) -> None:
        """Check activation magnitudes. TERMINATES if gates are dying."""
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'qk_proj_layer'):
                magnitude = block.qk_proj_layer.last_input_magnitude.item()

                if magnitude < self.threshold:
                    self.steps_below_threshold += 1
                    print(f"  WARNING: Step {step}: Layer {i} activation {magnitude:.4f} < {self.threshold} "
                          f"({self.steps_below_threshold}/{self.patience} steps)")

                    if self.steps_below_threshold >= self.patience:
                        print(f"\n  KILL SWITCH ACTIVATED: Activations below {self.threshold} "
                              f"for {self.patience} consecutive steps. Gates are dead.")
                        sys.exit(1)
                else:
                    self.steps_below_threshold = 0  # Reset counter


class AlphaGuardrails:
    """
    PRD Fix 10: Monitor M3 mixing alpha values.
    TERMINATES if alpha pins to extremes (frequency collapse).

    Per committee: "If alpha pins to 0.99 or 0.01, you've collapsed to single-memory mode."
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99, patience: int = 500):
        self.lower = lower
        self.upper = upper
        self.patience = patience
        self.steps_at_extreme = 0

    def check(self, model: nn.Module, step: int) -> None:
        """Check alpha values. TERMINATES if pinned at extreme."""
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'm3_mixer') and block.m3_mixer is not None:
                alpha = block.m3_mixer.alpha.item()

                if alpha < self.lower or alpha > self.upper:
                    self.steps_at_extreme += 1
                    print(f"  WARNING: Step {step}: Layer {i} alpha={alpha:.4f} at extreme "
                          f"({self.steps_at_extreme}/{self.patience} steps)")

                    if self.steps_at_extreme >= self.patience:
                        print(f"\n  KILL SWITCH ACTIVATED: Alpha pinned at {alpha:.4f} for "
                              f"{self.patience} steps. Frequency collapse detected.")
                        sys.exit(1)
                else:
                    self.steps_at_extreme = 0  # Reset counter


def check_m_init_gradient_health(model: nn.Module, threshold: float = 1e-6) -> bool:
    """
    PRD Fix 8: Verify M_init matrices are receiving gradients.

    Returns True if healthy, False if gradients are dead.
    Should be called after the first backward pass.
    """
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'M_init'):
            if block.M_init.grad is None:
                print(f"  ERROR: M_init gradient is None in layer {i}")
                return False

            grad_norm = block.M_init.grad.norm().item()
            if grad_norm < threshold:
                print(f"  ERROR: M_init gradient dead in layer {i}: grad_norm={grad_norm:.2e} < {threshold:.2e}")
                return False

            print(f"  Layer {i} M_init grad_norm: {grad_norm:.6f}")

    return True


def check_m3_orthogonality(model: nn.Module, threshold: float = 0.9) -> tuple:
    """
    PRD Fix 9: Check that global and local memory are learning different things.

    Returns (is_orthogonal, max_cos_sim). If cos_sim > threshold, gradients are
    too parallel (frequency collapse risk).
    """
    max_cos_sim = 0.0

    for i, block in enumerate(model.blocks):
        # Check if block has both global and local memory components
        if hasattr(block, 'M_init') and hasattr(block, 'mem_proj'):
            m_init_grad = block.M_init.grad
            mem_proj_grad = block.mem_proj.weight.grad if hasattr(block.mem_proj, 'weight') else None

            if m_init_grad is not None and mem_proj_grad is not None:
                # Compute cosine similarity between gradient directions
                cos_sim = F.cosine_similarity(
                    m_init_grad.flatten().unsqueeze(0),
                    mem_proj_grad.flatten().unsqueeze(0)
                ).abs().item()

                max_cos_sim = max(max_cos_sim, cos_sim)

                if cos_sim > threshold:
                    print(f"  WARNING: Layer {i} frequency collapse risk: cos_sim={cos_sim:.4f} > {threshold}")

    return max_cos_sim <= threshold, max_cos_sim


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    total_tokens: int,
    config: Union[AtlasConfig, Dict[str, Any]],
    loss: float,
    memory_states: Optional[list] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save checkpoint atomically (write to temp, then rename).

    Checkpoint contains everything needed for:
    - Resuming training
    - Running standalone evaluation
    - Analyzing training trajectory

    Args:
        checkpoint_dir: Directory for checkpoints
        model: The model
        optimizer: The optimizer
        step: Current training step
        total_tokens: Total tokens processed
        config: Training config
        loss: Current loss
        memory_states: Optional memory states (for exact resumption)
        extra_metadata: Additional info to save
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Convert AtlasConfig to dict for checkpoint portability
    config_dict = config.model_dump() if isinstance(config, AtlasConfig) else config

    checkpoint = {
        # Model state
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),

        # Training progress
        'step': step,
        'total_tokens': total_tokens,
        'loss': loss,

        # Config (for standalone eval - stored as dict for portability)
        'config': config_dict,

        # Metadata
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            **(extra_metadata or {}),
        },
    }

    # Optionally save memory states (can be large)
    if memory_states is not None:
        # Detach and move to CPU for storage
        checkpoint['memory_states'] = [
            {k: v.detach().cpu() if torch.is_tensor(v) else v
             for k, v in state.items()} if state else None
            for state in memory_states
        ]

    # Atomic write: save to temp file, then rename
    checkpoint_path = checkpoint_dir / f"step_{step:08d}.pt"
    temp_path = checkpoint_dir / f".tmp_step_{step:08d}.pt"

    torch.save(checkpoint, temp_path)
    shutil.move(str(temp_path), str(checkpoint_path))

    # Also save a 'latest' symlink for easy access
    latest_path = checkpoint_dir / "latest.pt"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(checkpoint_path.name)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu'),
):
    """
    Load checkpoint for resumption or evaluation.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to

    Returns:
        Dict with step, total_tokens, loss, config, metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'step': checkpoint['step'],
        'total_tokens': checkpoint['total_tokens'],
        'loss': checkpoint['loss'],
        'config': checkpoint.get('config'),
        'metadata': checkpoint.get('metadata', {}),
        'memory_states': checkpoint.get('memory_states'),
    }


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # Check for 'latest' symlink
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest.resolve()

    # Otherwise find highest step number
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if checkpoints:
        return checkpoints[-1]

    return None


class AtlasModel(nn.Module):
    """
    Full Atlas model with multiple AtlasMAG blocks.

    This is the full model for language modeling.
    For testing components, use MinimalAtlasBlock instead.
    """

    def __init__(self, config: Union[AtlasConfig, Dict[str, Any]]):
        super().__init__()

        # Support both AtlasConfig and dict (for checkpoint loading)
        if isinstance(config, AtlasConfig):
            self.config = config.model_dump()
            model_cfg = config.model
        else:
            # Dict from checkpoint - no validation, assume valid
            self.config = config
            model_cfg = ModelConfig.model_validate(config['model'])

        self.d_model = model_cfg.d_model
        self.n_layers = model_cfg.n_layers
        self.vocab_size = model_cfg.vocab_size

        # Token embedding with proper scaling (1/sqrt(d_model) for stable logits)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        # Scale embedding weights - critical for stable training with tied weights
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / (self.d_model ** 0.5))

        # AtlasMAG blocks
        self.blocks = nn.ModuleList([
            AtlasMAGBlock(
                d_model=model_cfg.d_model,
                n_heads=model_cfg.n_heads,
                window_size=model_cfg.window_size,
                n_persistent=model_cfg.n_persistent,
                omega_context=model_cfg.omega_context,
                poly_degree=model_cfg.poly_degree,
                ns_iterations=model_cfg.ns_iterations,
                dropout=model_cfg.dropout,
                m3_alpha_target=model_cfg.m3_alpha_target,
                m3_alpha_start=model_cfg.m3_alpha_start,
                m3_warmup_steps=model_cfg.m3_warmup_steps,
            )
            for _ in range(model_cfg.n_layers)
        ])

        # Output projection (tied weights with embedding)
        self.out_proj = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Weight tying (reduces parameters, improves performance)
        self.out_proj.weight = self.embed.weight

        # Layer norm
        self.final_norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_states: Optional[list] = None,
        shard_boundary: bool = False,
    ):
        """
        Forward pass through full model.

        Args:
            input_ids: Token IDs [B, L] (Long tensor)
            memory_states: List of memory state dicts per layer
            shard_boundary: Whether at TNT shard boundary

        Returns:
            logits: Output logits [B, L, vocab_size]
            new_states: Updated memory states
            telemetry: Combined telemetry from all blocks
        """
        if memory_states is None:
            memory_states = [None] * self.n_layers

        new_states = []
        all_telemetry = {}

        # Embed tokens
        h = self.embed(input_ids)  # [B, L] -> [B, L, D]
        for i, block in enumerate(self.blocks):
            h, state, telemetry = block(h, memory_states[i], shard_boundary)
            new_states.append(state)

            # Collect telemetry
            for k, v in telemetry.items():
                all_telemetry[f'layer_{i}/{k}'] = v

        h = self.final_norm(h)
        logits = self.out_proj(h)

        return logits, new_states, all_telemetry

    def get_regularization_loss(self) -> torch.Tensor:
        """Get combined regularization loss from all blocks."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            total = total + block.get_regularization_loss()
        return total

    def initialize_from_seed(self, seed_weights: "SeedWeights") -> None:
        """
        Initialize memory blocks from seed model weights.

        This implements ad hoc level stacking from NL Paper Section 7.3.
        The seed weights provide a "warm start" that avoids cold start instabilities.

        Args:
            seed_weights: SeedWeights object with memory_inits for each layer
        """
        from src.seed_model_loader import SeedWeights, validate_seed_weights

        # Validate dimensions match
        validate_seed_weights(seed_weights, self.d_model, self.n_layers)

        print(f"Initializing {self.n_layers} memory blocks from seed: {seed_weights.source_model}")

        # Copy seed weights to each block's M_init
        for i, (block, M_seed) in enumerate(zip(self.blocks, seed_weights.memory_inits)):
            # Get current device and dtype
            device = block.M_init.device
            dtype = block.M_init.dtype

            # Copy seed weights to M_init (the learnable initial memory state)
            with torch.no_grad():
                block.M_init.copy_(M_seed.to(device=device, dtype=dtype))

            print(f"  Block {i}: M_init initialized from seed layer {seed_weights.source_layers[i % len(seed_weights.source_layers)]}")

        print("Seed initialization complete.")


def create_differential_optimizer(
    model: nn.Module,
    base_lr: float,
    differential_ratio: float,
    weight_decay: float,
):
    """
    Create optimizer with differential learning rates.

    v4.2: Global memory learns 100x slower than local (neocortex vs hippocampus).
    This is NOT frozen - continuous consolidation still happens.

    v4.3 FIX: QK projection gain/scale params get 100x HIGHER LR.
    These params have vanishing gradients (~1e-7) due to deep path through
    Atlas memory system. Without boosted LR, they effectively don't learn.

    Args:
        model: The model
        base_lr: Base learning rate
        differential_ratio: Global LR = base_lr * ratio (e.g., 0.01 = 100x slower)
        weight_decay: Weight decay for AdamW

    Returns:
        AdamW optimizer with differential param groups
    """
    # Identify parameters by their role
    global_params = []
    local_params = []
    qk_gain_params = []  # v4.3: Separate group for QK projection gain (needs high LR)
    other_params = []

    for name, param in model.named_parameters():
        if 'qk_proj_layer.log_gain' in name or 'qk_proj_layer.output_scale' in name:
            # v4.3 FIX: QK projection learnable gain/scale need boosted LR
            # Gradient magnitude is ~1e-7 (vanishing through deep memory path)
            # v4.4: Now using log_gain (log-parameterized) for BF16/FP8 compatibility
            qk_gain_params.append(param)
        elif 'M_init' in name or 'S_init' in name or 'P_init' in name:
            # Memory state initializations are "global"
            global_params.append(param)
        elif 'm3_mixer' in name or 'alpha' in name:
            # M3 mixer controls state inheritance
            global_params.append(param)
        elif 'mem_' in name or 'memory' in name:
            # Memory-related parameters
            local_params.append(param)
        else:
            other_params.append(param)

    # v4.3: QK gain LR = 100x base (to compensate for ~1e-7 gradient magnitude)
    qk_gain_lr_multiplier = 100.0

    param_groups = [
        {
            'params': global_params,
            'lr': base_lr * differential_ratio,
            'name': 'global_memory',
        },
        {
            'params': local_params,
            'lr': base_lr,
            'name': 'local_memory',
        },
        {
            'params': qk_gain_params,
            'lr': base_lr * qk_gain_lr_multiplier,
            'name': 'qk_gain',  # v4.3: Boosted LR for vanishing gradient fix
        },
        {
            'params': other_params,
            'lr': base_lr,
            'name': 'other',
        },
    ]

    print("  Optimizer param groups:")
    print(f"    global_memory: {len(global_params)} params, lr={base_lr * differential_ratio:.6f}")
    print(f"    local_memory:  {len(local_params)} params, lr={base_lr:.6f}")
    print(f"    qk_gain:       {len(qk_gain_params)} params, lr={base_lr * qk_gain_lr_multiplier:.6f} (100x boost)")
    print(f"    other:         {len(other_params)} params, lr={base_lr:.6f}")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def training_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    memory_states: list,
    optimizer: torch.optim.Optimizer,
    cache_manager: HybridCacheManager,
    shard_boundary: bool,
    config: AtlasConfig,
) -> Dict[str, Any]:
    """
    Single training step with v4.2 safety features.

    Args:
        model: The Atlas model
        input_ids: Input token IDs [B, L]
        labels: Target token IDs [B, L]
        memory_states: Current memory states
        optimizer: Optimizer
        cache_manager: Cache manager for staleness detection
        shard_boundary: Whether at TNT shard boundary
        config: Training config (validated AtlasConfig)

    Returns:
        Dict with loss, metrics, new_states
    """
    # Forward pass
    logits, new_states, telemetry = model(
        input_ids,
        memory_states=memory_states,
        shard_boundary=shard_boundary,
    )

    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )

    # Add regularization (v4.2 safety)
    reg_loss = model.get_regularization_loss()
    total_loss = loss + 0.1 * reg_loss

    # Backward pass
    total_loss.backward()

    # Gradient clipping - REQUIRED field, no fallback
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

    # PRD Rev 4: Capture M_init gradient norms BEFORE zero_grad()
    m_init_grad_norms = {}
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'M_init') and block.M_init.grad is not None:
            m_init_grad_norms[f'layer_{i}'] = block.M_init.grad.norm().item()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # v4.2 CRITICAL: Invalidate cache IMMEDIATELY after weight update
    cache_manager.on_optimizer_step()

    # Check for kill switch
    if telemetry.get('KILL'):
        raise RuntimeError(telemetry['kill_reason'])

    return {
        'loss': loss.item(),
        'reg_loss': reg_loss.item(),
        'new_states': new_states,
        'telemetry': telemetry,
        'm_init_grad_norms': m_init_grad_norms,  # PRD Rev 4: For gradient health check
    }


def main():
    parser = argparse.ArgumentParser(description='Atlas with TNT Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--skip-isolation', action='store_true',
                        help='Skip isolation suite (NOT RECOMMENDED)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path (or "auto" for latest)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Pre-tokenized data directory (with train.npy, val.npy)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (default: use CUDA_VISIBLE_DEVICES or 0)')
    args = parser.parse_args()

    # GPU selection
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Using GPU {args.gpu}")

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(config_path)

    print("=" * 70)
    print("Atlas with TNT Training - PRD v4.2")
    print("=" * 70)
    print()

    # Run isolation suite first (unless skipped)
    if config.validation.isolation_suite and not args.skip_isolation:
        print("Running isolation suite first...")
        print("(Use --skip-isolation to skip, NOT RECOMMENDED)")
        print()

        # Import and run isolation suite
        from tests.test_isolation import run_isolation_suite

        if not run_isolation_suite():
            print()
            print("ISOLATION SUITE FAILED - ABORTING TRAINING")
            print("Fix the failing tests before proceeding.")
            sys.exit(1)

        print()
        print("Isolation suite passed. Proceeding to training...")
        print()

    # Setup device
    device = torch.device(config.hardware.device)
    dtype = getattr(torch, config.hardware.dtype)

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print()

    # Create model
    print("Creating model...")
    model = AtlasModel(config).to(device=device, dtype=dtype)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # Seed Model Initialization (NL Paper Section 7.3 - Ad Hoc Level Stacking)
    # ═══════════════════════════════════════════════════════════════════════════
    seed_initialized = False
    if config.seed_model.enabled:
        print("Loading seed model for memory initialization...")
        from src.seed_model_loader import load_seed_weights

        try:
            seed_weights = load_seed_weights(
                config=config.seed_model,
                target_d_model=config.model.d_model,
                n_layers=config.model.n_layers,
                device=device,
            )
            model.initialize_from_seed(seed_weights)
            seed_initialized = True
            print()
        except Exception as e:
            print(f"WARNING: Failed to load seed model: {e}")
            print("Continuing with random initialization...")
            print()
    else:
        print("Seed model disabled - using random initialization")
        print("(Enable in config: seed_model.enabled: true)")
        print()

    # Create optimizer with differential LR
    base_lr = config.training.lr.peak
    weight_decay = config.training.weight_decay

    # Use seed_lr_multiplier for global params when seed model is enabled
    # This keeps seeded weights close to their initialized values
    if seed_initialized:
        differential_ratio = config.seed_model.seed_lr_multiplier
        print(f"Using seed_lr_multiplier={differential_ratio} for global memory")
    else:
        differential_ratio = config.training.differential_lr_ratio

    optimizer = create_differential_optimizer(
        model,
        base_lr=base_lr,
        differential_ratio=differential_ratio,
        weight_decay=weight_decay,
    )

    # Print LR per group
    print("Learning rates:")
    for group in optimizer.param_groups:
        print(f"  {group['name']}: {group['lr']:.2e}")
    print()

    # Create cache manager - REQUIRED fields, no fallbacks
    cache_manager = HybridCacheManager(
        global_lr=base_lr * differential_ratio,
        local_lr=base_lr,
        lazy_threshold=config.cache.lazy_threshold,
        gradient_magnitude_threshold=config.cache.gradient_magnitude_threshold,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Data Loading
    # ═══════════════════════════════════════════════════════════════════════════
    print("Setting up data loading...")

    from src.data_loader import (
        get_tokenizer, create_dataloader, chinchilla_optimal_tokens,
        PreTokenizedDataset
    )

    # Get tokenizer - REQUIRED field
    tokenizer_name = config.data.tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = get_tokenizer(tokenizer_name)
    print(f"Vocabulary size: {len(tokenizer)}")

    # Calculate Chinchilla-optimal tokens
    optimal_tokens = chinchilla_optimal_tokens(n_params)
    actual_tokens = config.training.stage1.total_tokens
    print(f"Chinchilla-optimal tokens: {optimal_tokens:,}")
    print(f"Configured tokens: {actual_tokens:,}")

    # Determine data source
    data_dir = args.data_dir or config.data.pretokenized_path

    if data_dir and Path(data_dir).exists():
        # Use pre-tokenized data (faster)
        data_dir = Path(data_dir)
        train_path = data_dir / "train.npy"

        if train_path.exists():
            print(f"Using pre-tokenized data from: {data_dir}")

            # Load metadata if available
            metadata_path = data_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    data_metadata = json.load(f)
                print(f"  Train tokens: {data_metadata.get('train_tokens', 'unknown'):,}")
                print(f"  Val tokens: {data_metadata.get('val_tokens', 'unknown'):,}")

            dataset = PreTokenizedDataset(
                data_path=str(train_path),
                chunk_size=config.training.stage1.chunk_size,
                shard_length=config.training.stage1.shard_length,
            )

            from src.data_loader import tnt_collate_fn
            dataloader = DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=False,  # TNT requires sequential processing
                num_workers=4,  # Can use workers with map-style dataset
                collate_fn=tnt_collate_fn,
                pin_memory=True,
            )
        else:
            print(f"WARNING: Pre-tokenized data not found at {train_path}")
            print("Falling back to streaming...")
            data_dir = None

    if not data_dir or not Path(data_dir).exists():
        # Stream from HuggingFace - REQUIRED fields
        dataset_name = config.data.dataset
        dataset_subset = config.data.subset
        print(f"Streaming from: {dataset_name}" + (f" ({dataset_subset})" if dataset_subset else ""))

        dataloader = create_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            config=config,
            split="train",
            subset=dataset_subset,
        )

    print()
    print("Ready to train!")
    print("=" * 70)
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # Resume from checkpoint (if requested)
    # ═══════════════════════════════════════════════════════════════════════════
    checkpoint_dir = Path(__file__).parent.parent / config.paths.checkpoint_dir
    checkpoint_interval = config.training.checkpoint_interval

    start_step = 0
    start_tokens = 0

    if args.resume:
        if args.resume == "auto":
            resume_path = find_latest_checkpoint(checkpoint_dir)
        else:
            resume_path = Path(args.resume)

        if resume_path and resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            ckpt_info = load_checkpoint(resume_path, model, optimizer, device)
            start_step = ckpt_info['step']
            start_tokens = ckpt_info['total_tokens']
            print(f"  Resuming from step {start_step}, {start_tokens:,} tokens")
            print(f"  Last loss: {ckpt_info['loss']:.4f}")
            print()
        else:
            print(f"WARNING: Checkpoint not found at {args.resume}, starting fresh")
            print()

    # ═══════════════════════════════════════════════════════════════════════════
    # Training Loop
    # ═══════════════════════════════════════════════════════════════════════════

    # Initialize memory states
    memory_states = [None] * config.model.n_layers

    # Training stats - REQUIRED fields, no fallbacks
    step = start_step
    total_tokens = start_tokens
    chunk_size = config.training.stage1.chunk_size
    batch_size = config.training.batch_size
    log_interval = config.logging.telemetry.log_interval
    max_steps = actual_tokens // (batch_size * chunk_size)

    print(f"Max steps: {max_steps:,}")
    print(f"Tokens per step: {batch_size * chunk_size:,}")
    print(f"Checkpoint interval: {checkpoint_interval} steps")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # PRD Rev 4: Kill Switches Initialization
    # ═══════════════════════════════════════════════════════════════════════════
    print("Initializing kill switches (PRD Rev 4)...")
    activation_kill_switch = ActivationKillSwitch(threshold=0.1, patience=1000)
    alpha_guardrails = AlphaGuardrails(lower=0.01, upper=0.99, patience=500)
    m_init_checked = False  # Track if we've checked M_init gradients after step 0
    print("  - ActivationKillSwitch: threshold=0.1, patience=1000")
    print("  - AlphaGuardrails: range=[0.01, 0.99], patience=500")
    print("  - M_init gradient check: after step 0 (when M_init is used)")
    print("  - M3 orthogonality check: every 1000 steps")
    print()

    start_time = time.time()
    running_loss = 0.0
    last_loss = 0.0

    try:
        for batch_idx, batch in enumerate(dataloader):
            # Skip batches if resuming
            if batch_idx < start_step:
                continue

            # Get batch data
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            shard_boundary = batch['shard_boundary']

            if shard_boundary:
                print(f"  Step {step}: SHARD BOUNDARY (memory reset with M3 mixing)")

            # Training step
            try:
                result = training_step(
                    model, input_ids, labels, memory_states,
                    optimizer, cache_manager, shard_boundary, config
                )
                memory_states = result['new_states']
                running_loss += result['loss']
                last_loss = result['loss']

                # Logging
                if step % log_interval == 0:
                    avg_loss = running_loss / max(1, min(step - start_step, log_interval))
                    elapsed = time.time() - start_time
                    tokens_since_start = total_tokens - start_tokens
                    tokens_per_sec = tokens_since_start / max(1, elapsed)

                    print(f"  Step {step:6d}/{max_steps:6d} | "
                          f"loss={result['loss']:.4f} | "
                          f"avg_loss={avg_loss:.4f} | "
                          f"tokens/s={tokens_per_sec:.0f}")

                    running_loss = 0.0

                    # Check telemetry for warnings
                    for key, value in result['telemetry'].items():
                        if 'warning' in key.lower() or 'saturation' in key.lower():
                            print(f"    {key}={value}")

                # ═══════════════════════════════════════════════════════════════════
                # PRD Rev 4: Kill Switch Checks
                # ═══════════════════════════════════════════════════════════════════

                # Check 1: M_init gradient health (after FIRST step, when M_init is used)
                # Step 0 uses M_init because memory_states starts as [None]*n_layers
                # After step 0, memory states are passed forward (detached), M_init not used
                # Use gradient norms captured BEFORE zero_grad() in training_step()
                if step == 0 and not m_init_checked:
                    print("  Checking M_init gradient health...")
                    grad_norms = result.get('m_init_grad_norms', {})

                    if not grad_norms:
                        print("  ERROR: No M_init gradient norms captured")
                        print("\n  KILL SWITCH ACTIVATED: M_init is not receiving gradients")
                        sys.exit(1)

                    # Check each layer's gradient norm
                    all_healthy = True
                    for layer_name, norm in grad_norms.items():
                        if norm < 1e-6:
                            print(f"  ERROR: {layer_name} M_init grad_norm={norm:.2e} < 1e-6 (dead)")
                            all_healthy = False
                        else:
                            print(f"  {layer_name} M_init grad_norm: {norm:.6f}")

                    if not all_healthy:
                        print("\n  KILL SWITCH ACTIVATED: M_init gradients are dead")
                        sys.exit(1)

                    m_init_checked = True
                    print("  M_init gradient health: OK")

                # Check 2: Activation kill switch (every 100 steps)
                if step % 100 == 0:
                    activation_kill_switch.check(model, step)

                # Check 3: Alpha guardrails (every 100 steps)
                if step % 100 == 0:
                    alpha_guardrails.check(model, step)

                # Check 4: M3 orthogonality (every 1000 steps)
                if step % 1000 == 0 and step > 0:
                    is_orthogonal, max_cos_sim = check_m3_orthogonality(model)
                    if not is_orthogonal:
                        print(f"    M3 orthogonality WARNING: max_cos_sim={max_cos_sim:.4f}")

                # Checkpointing
                if step > 0 and step % checkpoint_interval == 0:
                    ckpt_path = save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        total_tokens=total_tokens,
                        config=config,
                        loss=last_loss,
                        extra_metadata={
                            'seed_initialized': seed_initialized,
                            'seed_model': config.seed_model.model_name if seed_initialized else None,
                        },
                    )
                    print(f"  Checkpoint saved: {ckpt_path.name}")

            except RuntimeError as e:
                if 'KILL' in str(e):
                    print(f"\n  KILL SWITCH TRIGGERED: {e}")
                    # Save emergency checkpoint before exit
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        total_tokens=total_tokens,
                        config=config,
                        loss=last_loss,
                        extra_metadata={
                            'kill_reason': str(e),
                            'seed_initialized': seed_initialized,
                            'seed_model': config.seed_model.model_name if seed_initialized else None,
                        },
                    )
                    sys.exit(1)
                raise

            step += 1
            total_tokens += batch_size * chunk_size

            # Check if done
            if step >= max_steps:
                print(f"\nReached max steps ({max_steps})")
                break

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at step {step}")
        # Save interrupt checkpoint
        print("Saving interrupt checkpoint...")
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            step=step,
            total_tokens=total_tokens,
            config=config,
            loss=last_loss,
            extra_metadata={
                'interrupted': True,
                'seed_initialized': seed_initialized,
                'seed_model': config.seed_model.model_name if seed_initialized else None,
            },
        )

    # Save final checkpoint
    if step > start_step:
        print("Saving final checkpoint...")
        final_ckpt = save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            step=step,
            total_tokens=total_tokens,
            config=config,
            loss=last_loss,
            extra_metadata={
                'final': True,
                'seed_initialized': seed_initialized,
                'seed_model': config.seed_model.model_name if seed_initialized else None,
            },
        )
        print(f"Final checkpoint: {final_ckpt}")

    # Final stats
    elapsed = time.time() - start_time
    tokens_since_start = total_tokens - start_tokens
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total steps: {step:,} (this run: {step - start_step:,})")
    print(f"Total tokens: {total_tokens:,} (this run: {tokens_since_start:,})")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Tokens/second: {tokens_since_start / max(1, elapsed):.0f}")


if __name__ == "__main__":
    main()
