"""
Seed Model Loader - Ad Hoc Level Stacking (NL Paper Section 7.3)

This module handles loading pre-trained models and extracting weights
to initialize the Atlas memory blocks, solving the "cold start" problem.

From the Nested Learning paper:
    "You initialize the continuum memory system, the CMS blocks, with weights
    from a small pre-trained MLP. It gives the model a kind of rudimentary map
    before training even starts."

Without seed initialization, random weights guarantee instability pockets
during early training. The seed provides a "warm start" that allows the
model to learn immediately instead of spending compute finding basic features.

Usage:
    from seed_model_loader import load_seed_weights, SeedWeights

    seed_weights = load_seed_weights(config.seed_model)
    # seed_weights.memory_inits[i] -> tensor for layer i's M_init
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

if TYPE_CHECKING:
    from .config_schema import SeedModelConfig

# Lazy import for transformers
_transformers = None


def _get_transformers() -> Any:
    """Lazy import transformers library."""
    global _transformers
    if _transformers is None:
        try:
            import transformers
            _transformers = transformers
        except ImportError:
            raise ImportError(
                "transformers library required for seed model loading. "
                "Install with: pip install transformers"
            )
    return _transformers


@dataclass
class SeedWeights:
    """
    Container for extracted seed weights.

    Attributes:
        memory_inits: List of tensors for initializing M_init in each Atlas block
        source_model: Name of the source model
        source_layers: Which layers were extracted
        d_model: Hidden dimension of the weights
    """
    memory_inits: List[torch.Tensor]
    source_model: str
    source_layers: List[int]
    d_model: int

    def __repr__(self) -> str:
        return (
            f"SeedWeights(source={self.source_model}, "
            f"layers={self.source_layers}, d_model={self.d_model}, "
            f"n_blocks={len(self.memory_inits)})"
        )


def extract_mlp_weights(
    model: nn.Module,
    layer_idx: int,
    target_d_model: int,
) -> torch.Tensor:
    """
    Extract and transform MLP weights from a single layer.

    SmolLM/Llama MLP structure:
        gate_proj: [intermediate_size, hidden_size]
        up_proj:   [intermediate_size, hidden_size]
        down_proj: [hidden_size, intermediate_size]

    We combine these to create a [d_model, d_model] matrix suitable
    for initializing the Atlas memory matrix M.

    Strategy: Use down_proj @ gate_proj to get [hidden_size, hidden_size]
    gate_proj is [intermediate, hidden], down_proj is [hidden, intermediate]
    Result: [hidden, intermediate] @ [intermediate, hidden] = [hidden, hidden]
    This captures the "compression" behavior of the MLP.

    Args:
        model: The loaded HuggingFace model
        layer_idx: Which layer to extract from
        target_d_model: Expected d_model (for validation)

    Returns:
        Tensor of shape [d_model, d_model] for M_init
    """
    # Access the MLP weights (works for Llama-style models)
    # Type ignores needed for dynamic HuggingFace model structure
    try:
        layer = model.model.layers[layer_idx]  # type: ignore[union-attr,index]
        mlp = layer.mlp  # type: ignore[union-attr]

        # Get weight tensors
        gate_proj: torch.Tensor = mlp.gate_proj.weight.data  # [intermediate, hidden]
        down_proj: torch.Tensor = mlp.down_proj.weight.data  # [hidden, intermediate]

        hidden_size = down_proj.shape[0]
        intermediate_size = down_proj.shape[1]

        # Validate dimensions
        if hidden_size != target_d_model:
            raise ValueError(
                f"Seed model hidden_size ({hidden_size}) doesn't match "
                f"target d_model ({target_d_model}). "
                f"Ensure config.model.d_model matches the seed model."
            )

        # Create memory-like matrix: down_proj @ gate_proj
        # gate_proj: [intermediate, hidden] = [1536, 576]
        # down_proj: [hidden, intermediate] = [576, 1536]
        # Result: [576, 1536] @ [1536, 576] = [576, 576] = [d_model, d_model]
        # This captures the MLP's learned compression/expansion patterns
        M_seed = down_proj @ gate_proj  # [hidden, intermediate] @ [intermediate, hidden]

        # Normalize to prevent scale explosion
        # Scale to have similar magnitude to typical M_init (randn * 0.01)
        M_seed = M_seed / (torch.norm(M_seed, p='fro') + 1e-7)
        M_seed = M_seed * 0.1  # Scale to reasonable initialization magnitude

        result: torch.Tensor = M_seed
        return result

    except AttributeError as e:
        raise ValueError(
            f"Could not access MLP weights at layer {layer_idx}. "
            f"Model structure may not be Llama-compatible. Error: {e}"
        )


def load_seed_weights(
    config: "SeedModelConfig",
    target_d_model: int,
    n_layers: int,
    device: torch.device = torch.device('cpu'),
) -> SeedWeights:
    """
    Load seed model and extract weights for Atlas memory initialization.

    Args:
        config: SeedModelConfig with model_name, source_layers, etc.
        target_d_model: The d_model of the Atlas model (for validation)
        n_layers: Number of Atlas layers to initialize
        device: Device to load weights to

    Returns:
        SeedWeights containing M_init tensors for each layer

    Raises:
        ValueError: If dimensions don't match or model can't be loaded
    """
    if not config.enabled:
        raise ValueError("Seed model is disabled in config")

    transformers = _get_transformers()

    print(f"Loading seed model: {config.model_name}")

    # Load the model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        torch_dtype=torch.float32,  # Load in fp32 for precision
        trust_remote_code=True,
    )

    # Get model config for validation
    model_config = model.config
    seed_hidden_size = model_config.hidden_size
    seed_n_layers = model_config.num_hidden_layers

    print(f"  Seed model: hidden_size={seed_hidden_size}, n_layers={seed_n_layers}")
    print(f"  Target Atlas: d_model={target_d_model}, n_layers={n_layers}")

    # Validate dimensions
    if seed_hidden_size != target_d_model:
        raise ValueError(
            f"Seed model hidden_size ({seed_hidden_size}) doesn't match "
            f"Atlas d_model ({target_d_model}). "
            f"Update config.model.d_model to {seed_hidden_size} to use this seed."
        )

    # Validate source layers exist
    for layer_idx in config.source_layers:
        if layer_idx >= seed_n_layers:
            raise ValueError(
                f"source_layer {layer_idx} >= seed model n_layers ({seed_n_layers})"
            )

    # Validate we have enough source layers for Atlas layers
    if len(config.source_layers) < n_layers:
        print(
            f"  Warning: {len(config.source_layers)} source layers for "
            f"{n_layers} Atlas layers. Will cycle through source layers."
        )

    # Extract weights for each Atlas layer
    memory_inits: List[torch.Tensor] = []

    for atlas_layer_idx in range(n_layers):
        # Cycle through source layers if we have fewer than n_layers
        source_idx = config.source_layers[atlas_layer_idx % len(config.source_layers)]

        print(f"  Extracting layer {source_idx} -> Atlas block {atlas_layer_idx}")

        M_init = extract_mlp_weights(model, source_idx, target_d_model)
        memory_inits.append(M_init.to(device))

    # Clean up model to free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Seed weights extracted: {n_layers} memory matrices of shape [{target_d_model}, {target_d_model}]")

    return SeedWeights(
        memory_inits=memory_inits,
        source_model=config.model_name,
        source_layers=config.source_layers,
        d_model=target_d_model,
    )


def get_random_init(d_model: int, n_layers: int, device: torch.device) -> SeedWeights:
    """
    Generate random initialization (for comparison/fallback).

    This produces the same initialization as the original random M_init,
    allowing A/B comparison between seeded and random starts.

    Args:
        d_model: Model dimension
        n_layers: Number of layers
        device: Device for tensors

    Returns:
        SeedWeights with random M_init tensors
    """
    memory_inits = [
        torch.randn(d_model, d_model, device=device) * 0.01
        for _ in range(n_layers)
    ]

    return SeedWeights(
        memory_inits=memory_inits,
        source_model="random",
        source_layers=[],
        d_model=d_model,
    )


def validate_seed_weights(
    seed_weights: SeedWeights,
    expected_d_model: int,
    expected_n_layers: int,
) -> None:
    """
    Validate that seed weights match expected dimensions.

    Args:
        seed_weights: The loaded seed weights
        expected_d_model: Expected model dimension
        expected_n_layers: Expected number of layers

    Raises:
        ValueError: If dimensions don't match
    """
    if seed_weights.d_model != expected_d_model:
        raise ValueError(
            f"Seed weights d_model ({seed_weights.d_model}) != "
            f"expected ({expected_d_model})"
        )

    if len(seed_weights.memory_inits) != expected_n_layers:
        raise ValueError(
            f"Seed weights n_layers ({len(seed_weights.memory_inits)}) != "
            f"expected ({expected_n_layers})"
        )

    for i, M in enumerate(seed_weights.memory_inits):
        if M.shape != (expected_d_model, expected_d_model):
            raise ValueError(
                f"Seed weight {i} has shape {M.shape}, "
                f"expected ({expected_d_model}, {expected_d_model})"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI for testing
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test seed model loading")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 5, 10, 15, 20, 25],
        help="Source layers to extract"
    )
    parser.add_argument(
        "--n-atlas-layers",
        type=int,
        default=6,
        help="Number of Atlas layers"
    )
    args = parser.parse_args()

    # Create a mock config
    from dataclasses import dataclass as dc

    @dc
    class MockConfig:
        enabled: bool = True
        model_name: str = args.model
        cache_dir: Optional[str] = None
        source_layers: List[int] = None  # type: ignore
        mapping_strategy: str = "mlp_to_memory"
        seed_lr_multiplier: float = 0.1

        def __post_init__(self) -> None:
            self.source_layers = args.layers

    config = MockConfig()

    print("=" * 60)
    print("Seed Model Loader Test")
    print("=" * 60)
    print()

    # SmolLM-135M has hidden_size=576
    target_d_model = 576

    seed_weights = load_seed_weights(
        config,  # type: ignore
        target_d_model=target_d_model,
        n_layers=args.n_atlas_layers,
    )

    print()
    print(f"Result: {seed_weights}")
    print()

    # Show statistics for each weight matrix
    for i, M in enumerate(seed_weights.memory_inits):
        print(f"  Layer {i}: shape={M.shape}, "
              f"mean={M.mean():.6f}, std={M.std():.6f}, "
              f"frobenius={torch.norm(M, p='fro'):.4f}")

    print()
    print("Validation passed!")
