"""
Data Loading for Atlas TNT Training - PRD v4.2

Implements:
- HuggingFace dataset streaming
- TNT-aware chunking (respects shard boundaries)
- Chinchilla-optimal data preparation

Supported datasets:
- FineWeb-Edu: High-quality educational web text
- SlimPajama: Diverse web crawl
- TinyStories: Simple stories for small models
- Custom: Any HuggingFace text dataset

Usage:
    from data_loader import create_dataloader, get_tokenizer

    tokenizer = get_tokenizer()
    dataloader = create_dataloader(
        dataset_name="HuggingFaceFW/fineweb-edu",
        tokenizer=tokenizer,
        config=config,
    )

    for batch in dataloader:
        # batch['input_ids']: [B, chunk_size]
        # batch['labels']: [B, chunk_size]
        # batch['shard_boundary']: bool
        ...
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, Any, Optional, Iterator, List, Union, TYPE_CHECKING
from pathlib import Path
import numpy as np

# Import config schema for type checking
if TYPE_CHECKING:
    from .config_schema import AtlasConfig

try:
    from .config_schema import AtlasConfig as _AtlasConfig  # noqa: F811
    _HAS_CONFIG_SCHEMA = True
except ImportError:
    _AtlasConfig = None  # type: ignore[misc,assignment]  # Intentional: optional dependency
    _HAS_CONFIG_SCHEMA = False

# Lazy imports for optional dependencies
_datasets = None
_transformers = None


def _get_datasets() -> Any:
    """Lazy import datasets library."""
    global _datasets
    if _datasets is None:
        try:
            import datasets
            _datasets = datasets
        except ImportError:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )
    return _datasets


def _get_transformers() -> Any:
    """Lazy import transformers library."""
    global _transformers
    if _transformers is None:
        try:
            import transformers
            _transformers = transformers
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
    return _transformers


def get_tokenizer(
    tokenizer_name: str = "gpt2",
    cache_dir: Optional[str] = None,
) -> Any:
    """
    Get a tokenizer for text processing.

    Args:
        tokenizer_name: HuggingFace tokenizer name
            - "gpt2": GPT-2 tokenizer (50257 vocab)
            - "meta-llama/Llama-2-7b-hf": Llama tokenizer (32000 vocab)
            - "EleutherAI/gpt-neox-20b": GPT-NeoX tokenizer
        cache_dir: Directory to cache tokenizer files

    Returns:
        Tokenizer instance
    """
    transformers = _get_transformers()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class TNTStreamingDataset(IterableDataset):
    """
    Streaming dataset for TNT training with shard boundary tracking.

    Key features:
    - Streams data from HuggingFace (no full download required)
    - Tracks shard boundaries for TNT resets
    - Efficient tokenization with caching
    - Supports multiple text columns

    Args:
        dataset_name: HuggingFace dataset name or path
        tokenizer: Tokenizer instance
        chunk_size: Tokens per training chunk
        shard_length: Tokens per TNT shard (reset interval)
        max_tokens: Maximum total tokens (for Chinchilla compliance)
        split: Dataset split ("train", "validation", etc.)
        text_column: Column name containing text
        streaming: Whether to stream (True) or download (False)
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        chunk_size: int = 128,
        shard_length: int = 2048,
        max_tokens: Optional[int] = None,
        split: str = "train",
        text_column: str = "text",
        streaming: bool = True,
        subset: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.shard_length = shard_length
        self.max_tokens = max_tokens
        self.text_column = text_column

        # Load dataset
        datasets = _get_datasets()

        load_kwargs = {
            "path": dataset_name,
            "split": split,
            "streaming": streaming,
        }
        if subset:
            load_kwargs["name"] = subset

        self.dataset = datasets.load_dataset(**load_kwargs)

        # State tracking
        self.token_buffer: List[int] = []
        self.total_tokens_yielded = 0
        self.position_in_shard = 0

    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize a single text string."""
        result: List[int] = self.tokenizer.encode(text, add_special_tokens=False)
        return result

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over chunks with shard boundary tracking.

        Yields dicts with:
            - input_ids: Token IDs [chunk_size]
            - labels: Shifted token IDs for LM loss [chunk_size]
            - shard_boundary: Whether this chunk starts a new shard
        """
        self.token_buffer = []
        self.total_tokens_yielded = 0
        self.position_in_shard = 0

        for example in self.dataset:
            # Check token limit
            if self.max_tokens and self.total_tokens_yielded >= self.max_tokens:
                return

            # Get text from example
            text = example.get(self.text_column, "")
            if not text:
                continue

            # Tokenize and add to buffer
            tokens = self._tokenize_text(text)
            self.token_buffer.extend(tokens)

            # Yield chunks while we have enough tokens
            while len(self.token_buffer) >= self.chunk_size + 1:
                # Check shard boundary
                shard_boundary = (self.position_in_shard == 0) and (self.total_tokens_yielded > 0)

                # Extract chunk
                chunk_tokens = self.token_buffer[:self.chunk_size + 1]
                self.token_buffer = self.token_buffer[self.chunk_size:]

                # Create input/label pairs (shifted by 1)
                input_ids = torch.tensor(chunk_tokens[:-1], dtype=torch.long)
                labels = torch.tensor(chunk_tokens[1:], dtype=torch.long)

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "shard_boundary": shard_boundary,
                }

                # Update counters
                self.total_tokens_yielded += self.chunk_size
                self.position_in_shard += self.chunk_size

                # Check for shard reset
                if self.position_in_shard >= self.shard_length:
                    self.position_in_shard = 0

                # Check token limit
                if self.max_tokens and self.total_tokens_yielded >= self.max_tokens:
                    return


class PreTokenizedDataset(Dataset):
    """
    Dataset for pre-tokenized data stored as numpy arrays.

    Useful for:
    - Faster loading after initial tokenization
    - Deterministic training (same data order)
    - Validation/test sets

    Args:
        data_path: Path to .npy file with token IDs
        chunk_size: Tokens per training chunk
        shard_length: Tokens per TNT shard
    """

    def __init__(
        self,
        data_path: str,
        chunk_size: int = 128,
        shard_length: int = 2048,
    ):
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.shard_length = shard_length

        # Load data
        self.tokens = np.load(self.data_path)
        self.n_tokens = len(self.tokens)

        # Calculate number of complete chunks
        self.n_chunks = (self.n_tokens - 1) // chunk_size

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Calculate position
        start = idx * self.chunk_size
        end = start + self.chunk_size + 1

        # Extract tokens
        chunk_tokens = self.tokens[start:end]

        # Calculate shard boundary
        global_position = idx * self.chunk_size
        position_in_shard = global_position % self.shard_length
        shard_boundary = (position_in_shard == 0) and (idx > 0)

        return {
            "input_ids": torch.tensor(chunk_tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk_tokens[1:], dtype=torch.long),
            "shard_boundary": shard_boundary,
        }


def create_dataloader(
    dataset_name: str,
    tokenizer: Any,
    config: Union["AtlasConfig", Dict[str, Any]],
    split: str = "train",
    subset: Optional[str] = None,
) -> DataLoader:
    """
    Create a DataLoader for TNT training.

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer instance
        config: Training config (AtlasConfig or dict for backwards compatibility)
        split: Dataset split
        subset: Dataset subset/config name

    Returns:
        DataLoader with TNT-aware batching
    """
    # Extract config values - support both AtlasConfig and dict
    if _HAS_CONFIG_SCHEMA and _AtlasConfig is not None and isinstance(config, _AtlasConfig):
        chunk_size = config.training.stage1.chunk_size
        shard_length = config.training.stage1.shard_length
        batch_size = config.training.batch_size
        max_tokens = config.training.stage1.total_tokens
    else:
        # Dict access for backwards compatibility
        config_dict = config if isinstance(config, dict) else {}
        chunk_size = config_dict['training']['stage1']['chunk_size']
        shard_length = config_dict['training']['stage1']['shard_length']
        batch_size = config_dict['training']['batch_size']
        max_tokens = config_dict['training']['stage1']['total_tokens']

    # Create dataset
    dataset = TNTStreamingDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        shard_length=shard_length,
        max_tokens=max_tokens,
        split=split,
        subset=subset,
    )

    # Create dataloader
    # Note: Streaming datasets don't support shuffle or num_workers > 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Streaming doesn't support shuffle
        num_workers=0,  # Streaming requires main process
        collate_fn=tnt_collate_fn,
        pin_memory=True,
    )

    return dataloader


def tnt_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for TNT training batches.

    Handles:
    - Stacking tensors
    - Aggregating shard boundary flags
    """
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])

    # Shard boundary is True if ANY item in batch is at boundary
    shard_boundary = any(x["shard_boundary"] for x in batch)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "shard_boundary": shard_boundary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-tokenization utilities
# ═══════════════════════════════════════════════════════════════════════════════

def pretokenize_dataset(
    dataset_name: str,
    tokenizer: Any,
    output_path: str,
    max_tokens: Optional[int] = None,
    split: str = "train",
    text_column: str = "text",
    subset: Optional[str] = None,
    show_progress: bool = True,
) -> int:
    """
    Pre-tokenize a HuggingFace dataset and save as numpy array.

    This is useful for:
    - Faster subsequent training runs
    - Exact reproducibility
    - Offline training

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer instance
        output_path: Path for output .npy file
        max_tokens: Maximum tokens to process
        split: Dataset split
        text_column: Column containing text
        subset: Dataset subset name
        show_progress: Whether to show progress bar

    Returns:
        Total number of tokens saved
    """
    datasets = _get_datasets()

    # Load dataset (streaming for memory efficiency)
    load_kwargs = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
    }
    if subset:
        load_kwargs["name"] = subset

    dataset = datasets.load_dataset(**load_kwargs)

    # Collect tokens
    all_tokens = []
    total_tokens = 0

    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(dataset, desc="Tokenizing")
        except ImportError:
            iterator = dataset
            print("Install tqdm for progress bar: pip install tqdm")
    else:
        iterator = dataset

    for example in iterator:
        if max_tokens and total_tokens >= max_tokens:
            break

        text = example.get(text_column, "")
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        total_tokens += len(tokens)

        if show_progress and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(tokens=f"{total_tokens:,}")

    # Trim to max_tokens if specified
    if max_tokens and len(all_tokens) > max_tokens:
        all_tokens = all_tokens[:max_tokens]

    # Save as numpy array
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tokens_array = np.array(all_tokens, dtype=np.int32)
    np.save(output_file, tokens_array)

    print(f"Saved {len(tokens_array):,} tokens to {output_file}")
    return len(tokens_array)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset recommendations
# ═══════════════════════════════════════════════════════════════════════════════

RECOMMENDED_DATASETS: Dict[str, Dict[str, Any]] = {
    "fineweb-edu": {
        "name": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",  # 10B token sample
        "description": "High-quality educational web text, curated for training",
        "text_column": "text",
    },
    "fineweb-edu-small": {
        "name": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-100BT",  # Smaller sample
        "description": "Smaller FineWeb-Edu sample for testing",
        "text_column": "text",
    },
    "slimpajama": {
        "name": "cerebras/SlimPajama-627B",
        "subset": None,
        "description": "Diverse web crawl, good for general language modeling",
        "text_column": "text",
    },
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "subset": None,
        "description": "Simple children's stories, good for small model validation",
        "text_column": "text",
    },
    "openwebtext": {
        "name": "openwebtext",
        "subset": None,
        "description": "Open recreation of WebText (GPT-2 training data)",
        "text_column": "text",
    },
}


def get_recommended_dataset(name: str) -> Dict[str, Any]:
    """Get recommended dataset config by short name."""
    if name not in RECOMMENDED_DATASETS:
        available = ", ".join(RECOMMENDED_DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return RECOMMENDED_DATASETS[name]


# ═══════════════════════════════════════════════════════════════════════════════
# Chinchilla scaling utilities
# ═══════════════════════════════════════════════════════════════════════════════

def chinchilla_optimal_tokens(n_params: int, ratio: float = 20.0) -> int:
    """
    Calculate Chinchilla-optimal training tokens.

    Chinchilla paper suggests ~20 tokens per parameter for compute-optimal training.

    Args:
        n_params: Number of model parameters
        ratio: Tokens-to-parameters ratio (default: 20)

    Returns:
        Optimal number of training tokens
    """
    return int(n_params * ratio)


def estimate_training_steps(
    total_tokens: int,
    batch_size: int,
    chunk_size: int,
    gradient_accumulation: int = 1,
) -> int:
    """
    Estimate total training steps.

    Args:
        total_tokens: Total tokens to train on
        batch_size: Batch size
        chunk_size: Tokens per chunk
        gradient_accumulation: Gradient accumulation steps

    Returns:
        Estimated training steps
    """
    tokens_per_step = batch_size * chunk_size * gradient_accumulation
    return total_tokens // tokens_per_step


if __name__ == "__main__":
    # Example usage
    print("Data Loading Module for Atlas TNT Training")
    print("=" * 50)
    print()
    print("Recommended datasets:")
    for name, info in RECOMMENDED_DATASETS.items():
        print(f"  {name}: {info['description']}")
    print()
    print("Example usage:")
    print("  tokenizer = get_tokenizer('gpt2')")
    print("  dataloader = create_dataloader('HuggingFaceFW/fineweb-edu', tokenizer, config)")
    print()

    # Chinchilla calculation for 50M model
    n_params = 50_000_000
    optimal_tokens = chinchilla_optimal_tokens(n_params)
    print(f"Chinchilla-optimal tokens for 50M params: {optimal_tokens:,}")
