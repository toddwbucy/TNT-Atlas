"""
Atlas with TNT Training - Implementation of PRD v4.2

Core components:
- PolynomialFeatureLayer: O(d^p) capacity expansion
- QKProjectionLayer: Query-Key alignment with tanh safety gate
- M3MixingState: Continuous state inheritance (no binary switches)
- HybridCacheManager: Cache slow, recompute fast
- AtlasMAGBlock: Main architecture block
- MinimalAtlasBlock: Simplified block for testing

Key principle: "Don't use binary switches - use continuous dynamics."
"""

from .polynomial import PolynomialFeatureLayer
from .qk_projection import QKProjectionLayer
from .m3_mixing import M3MixingState, reset_atlas_states
from .cache_manager import HybridCacheManager, get_omega_context_v4
from .newton_schulz import (
    newton_schulz_5,
    newton_schulz_k,
    get_newton_schulz_iterations,
    newton_schulz_with_telemetry,
)
from .atlas_memory import AtlasMemory, atlas_memory_update
from .attention import SlidingWindowAttention
from .atlas_block import AtlasMAGBlock
from .minimal_block import MinimalAtlasBlock, LinearTestModel
from .seed_model_loader import load_seed_weights, SeedWeights, get_random_init

__all__ = [
    # Polynomial features
    'PolynomialFeatureLayer',
    # Q-K projection
    'QKProjectionLayer',
    # M3 mixing
    'M3MixingState',
    'reset_atlas_states',
    # Cache management
    'HybridCacheManager',
    'get_omega_context_v4',
    # Newton-Schulz
    'newton_schulz_5',
    'newton_schulz_k',
    'get_newton_schulz_iterations',
    'newton_schulz_with_telemetry',
    # Atlas memory
    'AtlasMemory',
    'atlas_memory_update',
    # Attention
    'SlidingWindowAttention',
    # Main blocks
    'AtlasMAGBlock',
    'MinimalAtlasBlock',
    'LinearTestModel',
    # Seed model loading (NL Section 7.3)
    'load_seed_weights',
    'SeedWeights',
    'get_random_init',
]

__version__ = '0.1.0'
