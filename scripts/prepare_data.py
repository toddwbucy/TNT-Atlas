#!/usr/bin/env python3
"""
Data Preparation Script - Pre-tokenize with 80/20 Train/Val Split

This script:
1. Downloads dataset from HuggingFace (streaming)
2. Splits 80/20 for train/val
3. Tokenizes both splits
4. Saves as numpy arrays for fast loading

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
    python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --tokens 1B

Output:
    data/
    ├── train.npy          # Training tokens
    ├── val.npy            # Validation tokens
    └── metadata.json      # Split info, tokenizer, counts
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional
import hashlib

import numpy as np

# Add project root to path for package imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import get_tokenizer, _get_datasets, RECOMMENDED_DATASETS
from src.config_schema import load_config


def parse_token_count(s: str) -> int:
    """Parse token count strings like '1B', '100M', '1_000_000'."""
    s = s.upper().replace('_', '').replace(',', '')
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000, 'T': 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def prepare_data(
    dataset_name: str,
    output_dir: Path,
    tokenizer_name: str = "gpt2",
    total_tokens: int = 1_000_000_000,
    val_ratio: float = 0.2,
    subset: Optional[str] = None,
    text_column: str = "text",
    seed: int = 42,
    show_progress: bool = True,
):
    """
    Prepare train/val data splits.

    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Directory for output files
        tokenizer_name: Tokenizer to use
        total_tokens: Total tokens to collect
        val_ratio: Fraction for validation (default 0.2 = 20%)
        subset: Dataset subset/config
        text_column: Column containing text
        seed: Random seed for reproducibility
        show_progress: Show progress bar
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Atlas TNT - Data Preparation")
    print("=" * 70)
    print()
    print(f"Dataset: {dataset_name}" + (f" ({subset})" if subset else ""))
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Target tokens: {total_tokens:,}")
    print(f"Val ratio: {val_ratio:.0%}")
    print(f"Output: {output_dir}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(tokenizer_name)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size:,}")
    print()

    # Load dataset (streaming)
    print("Loading dataset (streaming)...")
    datasets = _get_datasets()

    load_kwargs = {
        "path": dataset_name,
        "split": "train",
        "streaming": True,
    }
    if subset:
        load_kwargs["name"] = subset

    dataset = datasets.load_dataset(**load_kwargs)

    # Set up random number generator for splitting
    rng = np.random.default_rng(seed)

    # Collect tokens with train/val split
    train_tokens = []
    val_tokens = []
    total_collected = 0
    docs_processed = 0

    print("Tokenizing and splitting...")
    start_time = time.time()

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_tokens, unit='tok', unit_scale=True) if show_progress else None
    except ImportError:
        pbar = None
        print("(Install tqdm for progress bar)")

    for example in dataset:
        if total_collected >= total_tokens:
            break

        text = example.get(text_column, "")
        if not text or len(text) < 50:  # Skip very short docs
            continue

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue

        # Decide train or val based on document (not token) level
        # This keeps related tokens together
        if rng.random() < val_ratio:
            val_tokens.extend(tokens)
        else:
            train_tokens.extend(tokens)

        total_collected = len(train_tokens) + len(val_tokens)
        docs_processed += 1

        if pbar:
            pbar.n = total_collected
            pbar.refresh()
        elif docs_processed % 10000 == 0:
            print(f"  Processed {docs_processed:,} docs, {total_collected:,} tokens...")

    if pbar:
        pbar.close()

    elapsed = time.time() - start_time

    # Trim to exact token counts if needed
    target_val = int(total_tokens * val_ratio)
    target_train = total_tokens - target_val

    if len(train_tokens) > target_train:
        train_tokens = train_tokens[:target_train]
    if len(val_tokens) > target_val:
        val_tokens = val_tokens[:target_val]

    print()
    print(f"Tokenization complete in {elapsed:.1f}s")
    print(f"  Documents processed: {docs_processed:,}")
    print(f"  Train tokens: {len(train_tokens):,}")
    print(f"  Val tokens: {len(val_tokens):,}")
    print(f"  Actual split: {len(train_tokens)/(len(train_tokens)+len(val_tokens)):.1%} / {len(val_tokens)/(len(train_tokens)+len(val_tokens)):.1%}")
    print()

    # Convert to numpy arrays
    print("Saving to numpy arrays...")

    # Use int32 for vocab < 2B (saves space vs int64)
    dtype = np.int32 if vocab_size < 2**31 else np.int64

    train_array = np.array(train_tokens, dtype=dtype)
    val_array = np.array(val_tokens, dtype=dtype)

    train_path = output_dir / "train.npy"
    val_path = output_dir / "val.npy"

    np.save(train_path, train_array)
    np.save(val_path, val_array)

    # Calculate checksums
    train_hash = hashlib.sha256(train_array.tobytes()).hexdigest()[:16]
    val_hash = hashlib.sha256(val_array.tobytes()).hexdigest()[:16]

    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "subset": subset,
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "total_tokens": len(train_tokens) + len(val_tokens),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "val_ratio": val_ratio,
        "seed": seed,
        "dtype": str(dtype),
        "train_file": "train.npy",
        "val_file": "val.npy",
        "train_sha256": train_hash,
        "val_sha256": val_hash,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print file sizes
    train_size_mb = train_path.stat().st_size / (1024 * 1024)
    val_size_mb = val_path.stat().st_size / (1024 * 1024)

    print()
    print("Files saved:")
    print(f"  {train_path}: {train_size_mb:.1f} MB ({len(train_tokens):,} tokens)")
    print(f"  {val_path}: {val_size_mb:.1f} MB ({len(val_tokens):,} tokens)")
    print(f"  {metadata_path}")
    print()
    print("=" * 70)
    print("Data preparation complete!")
    print("=" * 70)

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Prepare training data with train/val split')
    parser.add_argument('--config', type=str, help='Config file (uses data section)')
    parser.add_argument('--dataset', type=str, help='HuggingFace dataset name')
    parser.add_argument('--subset', type=str, help='Dataset subset/config')
    parser.add_argument('--tokens', type=str, default='1B', help='Total tokens (e.g., 1B, 100M)')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio (default: 0.2)')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Tokenizer name')
    parser.add_argument('--output', type=str, default='data/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Load from config if provided
    if args.config:
        config_path = Path(__file__).parent.parent / args.config
        # Use validated Pydantic config - fails fast if config is invalid
        config = load_config(config_path)

        # Use config values, allow CLI overrides
        dataset_name = args.dataset or config.data.dataset
        subset = args.subset or config.data.subset
        tokenizer_name = args.tokenizer if args.tokenizer != 'gpt2' else config.data.tokenizer
        total_tokens = parse_token_count(args.tokens) if args.tokens != '1B' else config.training.stage1.total_tokens
        output_dir = Path(args.output) if args.output != 'data/' else Path(config.paths.data_dir)
    else:
        # Check for shorthand dataset names
        dataset_name = args.dataset
        if dataset_name and dataset_name in RECOMMENDED_DATASETS:
            rec = RECOMMENDED_DATASETS[dataset_name]
            dataset_name = rec['name']
            subset = args.subset or rec.get('subset')
        else:
            subset = args.subset

        if not dataset_name:
            print("Error: Must provide --dataset or --config")
            sys.exit(1)

        tokenizer_name = args.tokenizer
        total_tokens = parse_token_count(args.tokens)
        output_dir = Path(args.output)

    prepare_data(
        dataset_name=dataset_name,
        output_dir=output_dir,
        tokenizer_name=tokenizer_name,
        total_tokens=total_tokens,
        val_ratio=args.val_ratio,
        subset=subset,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
