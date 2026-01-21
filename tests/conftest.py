"""
Shared pytest fixtures for Atlas TNT tests.
"""

from pathlib import Path
from typing import Dict, Any

import pytest
import torch
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_config_dict() -> Dict[str, Any]:
    """Minimal valid config dict for fast testing."""
    return {
        "model": {
            "vocab_size": 1000,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "window_size": 64,
            "n_persistent": 2,
            "omega_context": 4,
            "poly_degree": 2,
            "ns_iterations": 1,
            "m3_alpha_target": 0.5,
            "m3_alpha_start": 0.1,
            "m3_warmup_steps": 10,
            "qk_gain_circuit_breaker": 2.0,
            "saturation_kill_threshold": 0.20,
            "saturation_kill_patience": 10,
            "dropout": 0.0,
            "eta_scale": 0.1,
        },
        "training": {
            "stage1": {
                "shard_length": 128,
                "total_tokens": 10000,
                "chunk_size": 32,
            },
            "stage2": {
                "chunk_sizes": [2, 4],
                "tokens_per_chunk": 1000,
            },
            "optimizer": "AdamW",
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "lr": {
                "peak": 0.001,
                "min": 0.0001,
                "warmup_steps": 10,
                "schedule": "cosine",
            },
            "differential_lr_ratio": 0.01,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "checkpoint_interval": 100,
        },
        "cache": {
            "enabled": True,
            "lazy_threshold": 0,
            "gradient_magnitude_threshold": 1e-8,
        },
        "logging": {
            "level": "WARNING",
            "tensorboard": False,
            "wandb": False,
            "telemetry": {
                "log_interval": 10,
                "track_saturation": True,
                "track_alpha": True,
                "track_cache_stats": True,
            },
        },
        "hardware": {
            "device": "cpu",
            "dtype": "float32",
            "distributed": False,
        },
        "validation": {
            "isolation_suite": False,
            "max_nan_steps": 0,
            "max_loss_spike_ratio": 3.0,
            "min_alpha_range": [0.1, 0.9],
        },
        "paths": {
            "data_dir": "tests/fixtures/",
            "checkpoint_dir": "tests/fixtures/checkpoints/",
            "log_dir": "tests/fixtures/logs/",
            "tensorboard_dir": "tests/fixtures/tensorboard/",
        },
        "data": {
            "dataset": "test-dataset",
            "subset": None,
            "text_column": "text",
            "tokenizer": "gpt2",
            "pretokenized_path": None,
            "streaming": False,
        },
        "validation_run": {
            "gpu": 0,
            "max_batches": 10,
            "watch_interval": 60,
        },
        "m3_safety": {
            "danger_high": 0.95,
            "danger_low": 0.05,
            "post_warmup_monitor_steps": 10,
            "regularization_low_boundary": 0.1,
            "regularization_high_boundary": 0.9,
        },
        "newton_schulz": {
            "warmup_steps": 5,
            "k_max": 3,
            "k_min": 1,
        },
        "seed_model": {
            "enabled": False,  # Disabled for tests (no need to download SmolLM)
            "model_name": "HuggingFaceTB/SmolLM-135M",
            "cache_dir": None,
            "source_layers": [0, 1, 2, 3, 4, 5],
            "mapping_strategy": "mlp_to_memory",
            "seed_lr_multiplier": 0.1,
        },
    }


@pytest.fixture
def tiny_config_path(tmp_path: Path, tiny_config_dict: Dict[str, Any]) -> Path:
    """Write tiny config to a temp file and return path."""
    import yaml
    config_path = tmp_path / "tiny_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(tiny_config_dict, f)
    return config_path


@pytest.fixture
def tiny_config(tiny_config_path: Path):
    """Load and validate tiny config via Pydantic."""
    from src.config_schema import load_config
    return load_config(tiny_config_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_tokens() -> np.ndarray:
    """Small tokenized dataset for testing."""
    # Random token IDs in valid range
    np.random.seed(42)
    return np.random.randint(0, 1000, size=10000, dtype=np.int32)


@pytest.fixture
def tiny_tokens_path(tmp_path: Path, tiny_tokens: np.ndarray) -> Path:
    """Write tiny tokens to a temp file and return path."""
    tokens_path = tmp_path / "tiny_tokens.npy"
    np.save(tokens_path, tiny_tokens)
    return tokens_path


# ═══════════════════════════════════════════════════════════════════════════════
# Model Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_model(tiny_config):
    """Create a small AtlasModel for testing."""
    # Import here to avoid issues if model code has errors
    from train import AtlasModel
    return AtlasModel(tiny_config)


# ═══════════════════════════════════════════════════════════════════════════════
# Device Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device for tests that don't need GPU."""
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# Pytest Hooks
# ═══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "unit: Fast unit tests (no GPU, no network)")
    config.addinivalue_line("markers", "integration: Integration tests (may be slow)")
    config.addinivalue_line("markers", "gpu: Requires CUDA GPU")
    config.addinivalue_line("markers", "network: Requires network access")
    config.addinivalue_line("markers", "slow: Takes >5s to run")
    config.addinivalue_line("markers", "isolation: ML isolation tests")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests if no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
