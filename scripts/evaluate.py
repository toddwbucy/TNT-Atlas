#!/usr/bin/env python3
"""
Batch Evaluation Script - Run validation on checkpoints

This script runs independently from training, typically on a different GPU.
It processes a directory of checkpoints and generates validation metrics.

Key features:
- Processes multiple checkpoints in batch
- Runs on specified GPU (default: GPU 2 for validation, GPU 0/1 for training)
- Outputs JSONL results for plotting training curves
- Self-contained - uses config embedded in checkpoints

Usage:
    # Run validation on all checkpoints in directory
    python scripts/evaluate.py --checkpoint-dir runs/checkpoints/ --data-dir data/

    # Run validation on specific checkpoint
    python scripts/evaluate.py --checkpoint runs/checkpoints/step_10000.pt --data-dir data/

    # Run on specific GPU
    python scripts/evaluate.py --checkpoint-dir runs/checkpoints/ --data-dir data/ --gpu 2

    # Watch for new checkpoints (continuous mode)
    python scripts/evaluate.py --checkpoint-dir runs/checkpoints/ --data-dir data/ --watch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path for package imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ValidationResult:
    """Validation result for a single checkpoint."""
    checkpoint_path: str
    step: int
    total_tokens: int
    train_loss: float  # Loss at checkpoint time
    val_loss: float
    val_perplexity: float
    val_tokens_processed: int
    validation_time_seconds: float
    timestamp: str
    gpu_id: int
    metadata: Dict[str, Any]


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """
    Load model from checkpoint (uses config embedded in checkpoint).

    Returns:
        model: Loaded model in validation mode
        checkpoint_info: Dict with step, tokens, config, etc.
    """
    # Import here to avoid circular imports
    from train import AtlasModel

    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    model = AtlasModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.train(False)  # Set to validation mode

    return model, {
        'step': checkpoint['step'],
        'total_tokens': checkpoint['total_tokens'],
        'train_loss': checkpoint['loss'],
        'config': config,
        'metadata': checkpoint.get('metadata', {}),
    }


def run_validation(
    model: torch.nn.Module,
    val_data_path: Path,
    config: Dict[str, Any],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run validation on model using validation data.

    Args:
        model: Model in validation mode
        val_data_path: Path to val.npy
        config: Model config
        device: Device to run on
        max_batches: Maximum batches to process (None = all)

    Returns:
        Dict with val_loss, val_perplexity, tokens_processed
    """
    # Load validation data
    val_tokens = np.load(val_data_path)
    n_tokens = len(val_tokens)

    chunk_size = config['training']['stage1']['chunk_size']
    batch_size = config['training']['batch_size']  # Required field, no fallback
    shard_length = config['training']['stage1']['shard_length']  # Memory reset interval

    # Calculate number of chunks
    n_chunks = (n_tokens - 1) // chunk_size
    n_batches = n_chunks // batch_size

    if max_batches:
        n_batches = min(n_batches, max_batches)

    total_loss = 0.0
    total_tokens_processed = 0
    n_layers = config['model']['n_layers']

    # Track tokens since last shard boundary (for memory reset)
    tokens_since_shard_boundary = 0

    with torch.no_grad():
        memory_states = [None] * n_layers

        for batch_idx in range(n_batches):
            # Check if we need a shard boundary (memory reset)
            # This matches training behavior where memory resets every shard_length tokens
            shard_boundary = tokens_since_shard_boundary >= shard_length
            if shard_boundary:
                tokens_since_shard_boundary = 0

            # Build batch
            input_ids_list = []
            labels_list = []

            for b in range(batch_size):
                chunk_idx = batch_idx * batch_size + b
                if chunk_idx >= n_chunks:
                    break

                start = chunk_idx * chunk_size
                end = start + chunk_size + 1

                chunk = val_tokens[start:end]
                input_ids_list.append(torch.tensor(chunk[:-1], dtype=torch.long))
                labels_list.append(torch.tensor(chunk[1:], dtype=torch.long))

            if not input_ids_list:
                break

            input_ids = torch.stack(input_ids_list).to(device)
            labels = torch.stack(labels_list).to(device)

            # Forward pass (with shard boundaries matching training behavior)
            logits, memory_states, _ = model(
                input_ids,
                memory_states=memory_states,
                shard_boundary=shard_boundary,
            )

            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            batch_tokens = input_ids.size(0) * input_ids.size(1)
            total_loss += loss.item() * batch_tokens
            total_tokens_processed += batch_tokens
            tokens_since_shard_boundary += batch_tokens

    avg_loss = total_loss / max(1, total_tokens_processed)
    perplexity = np.exp(min(avg_loss, 100))  # Cap to avoid overflow

    return {
        'val_loss': avg_loss,
        'val_perplexity': perplexity,
        'tokens_processed': total_tokens_processed,
    }


def validate_checkpoint(
    checkpoint_path: Path,
    val_data_path: Path,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> ValidationResult:
    """
    Run validation on a single checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        val_data_path: Path to val.npy
        device: Device to run on
        max_batches: Max batches for validation (None = all)

    Returns:
        ValidationResult with all metrics
    """
    start_time = time.time()

    # Load model
    model, ckpt_info = load_model_from_checkpoint(checkpoint_path, device)

    # Run validation
    val_metrics = run_validation(
        model=model,
        val_data_path=val_data_path,
        config=ckpt_info['config'],
        device=device,
        max_batches=max_batches,
    )

    elapsed = time.time() - start_time

    # Get GPU ID from device
    gpu_id = device.index if device.type == 'cuda' and device.index is not None else 0

    return ValidationResult(
        checkpoint_path=str(checkpoint_path),
        step=ckpt_info['step'],
        total_tokens=ckpt_info['total_tokens'],
        train_loss=ckpt_info['train_loss'],
        val_loss=val_metrics['val_loss'],
        val_perplexity=val_metrics['val_perplexity'],
        val_tokens_processed=val_metrics['tokens_processed'],
        validation_time_seconds=elapsed,
        timestamp=datetime.now().isoformat(),
        gpu_id=gpu_id,
        metadata=ckpt_info.get('metadata', {}),
    )


def find_unvalidated_checkpoints(
    checkpoint_dir: Path,
    results_file: Path,
) -> List[Path]:
    """Find checkpoints that haven't been validated yet."""
    # Get all checkpoints
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))

    # Load existing results
    validated_paths = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    validated_paths.add(result['checkpoint_path'])
                except json.JSONDecodeError:
                    continue

    # Filter to unvalidated
    unvalidated = [
        ckpt for ckpt in checkpoints
        if str(ckpt) not in validated_paths and str(ckpt.resolve()) not in validated_paths
    ]

    return unvalidated


def main():
    parser = argparse.ArgumentParser(description='Run validation on checkpoints')
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint to validate')
    parser.add_argument('--checkpoint-dir', type=str, help='Directory of checkpoints')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory with val.npy')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL file (default: checkpoint_dir/validation_results.jsonl)')
    parser.add_argument('--gpu', type=int, default=2, help='GPU to use for validation (default: 2)')
    parser.add_argument('--max-batches', type=int, default=None, help='Max batches per validation (for quick testing)')
    parser.add_argument('--watch', action='store_true', help='Watch for new checkpoints continuously')
    parser.add_argument('--watch-interval', type=int, default=60, help='Seconds between watch checks (default: 60)')
    args = parser.parse_args()

    # Validate args
    if not args.checkpoint and not args.checkpoint_dir:
        parser.error("Must provide either --checkpoint or --checkpoint-dir")

    # Set GPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device('cuda')
        print(f"Using GPU {args.gpu}")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available, using CPU")

    # Validate data directory
    data_dir = Path(args.data_dir)
    val_path = data_dir / "val.npy"
    if not val_path.exists():
        print(f"ERROR: Validation data not found at {val_path}")
        print("Run scripts/prepare_data.py first to create train/val splits")
        sys.exit(1)

    # Load data metadata
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            data_metadata = json.load(f)
        print(f"Validation tokens: {data_metadata.get('val_tokens', 'unknown'):,}")
    print()

    # Determine output file
    if args.output:
        results_file = Path(args.output)
    elif args.checkpoint_dir:
        results_file = Path(args.checkpoint_dir) / "validation_results.jsonl"
    else:
        results_file = Path(args.checkpoint).parent / "validation_results.jsonl"

    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Single checkpoint mode
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        print(f"Validating: {checkpoint_path}")
        result = validate_checkpoint(
            checkpoint_path=checkpoint_path,
            val_data_path=val_path,
            device=device,
            max_batches=args.max_batches,
        )

        # Print results
        print(f"\n{'='*60}")
        print(f"Step: {result.step:,}")
        print(f"Tokens: {result.total_tokens:,}")
        print(f"Train Loss: {result.train_loss:.4f}")
        print(f"Val Loss: {result.val_loss:.4f}")
        print(f"Val Perplexity: {result.val_perplexity:.2f}")
        print(f"Validation Time: {result.validation_time_seconds:.1f}s")
        print(f"{'='*60}")

        # Save result
        with open(results_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')
        print(f"\nResult saved to: {results_file}")

        return

    # Batch/watch mode
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Results file: {results_file}")
    print()

    while True:
        # Find unvalidated checkpoints
        checkpoints = find_unvalidated_checkpoints(checkpoint_dir, results_file)

        if not checkpoints:
            if args.watch:
                print(f"No new checkpoints. Waiting {args.watch_interval}s...")
                time.sleep(args.watch_interval)
                continue
            else:
                print("No unvalidated checkpoints found.")
                break

        print(f"Found {len(checkpoints)} checkpoint(s) to validate")

        for i, ckpt_path in enumerate(checkpoints):
            print(f"\n[{i+1}/{len(checkpoints)}] Validating: {ckpt_path.name}")

            try:
                result = validate_checkpoint(
                    checkpoint_path=ckpt_path,
                    val_data_path=val_path,
                    device=device,
                    max_batches=args.max_batches,
                )

                # Print summary
                print(f"  Step {result.step:,}: "
                      f"train_loss={result.train_loss:.4f}, "
                      f"val_loss={result.val_loss:.4f}, "
                      f"ppl={result.val_perplexity:.2f} "
                      f"({result.validation_time_seconds:.1f}s)")

                # Append to results
                with open(results_file, 'a') as f:
                    f.write(json.dumps(asdict(result)) + '\n')

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        if not args.watch:
            break

        print(f"\nWaiting {args.watch_interval}s for new checkpoints...")
        time.sleep(args.watch_interval)

    print("\n" + "="*60)
    print("Validation complete!")
    print(f"Results saved to: {results_file}")
    print("="*60)


if __name__ == "__main__":
    main()
