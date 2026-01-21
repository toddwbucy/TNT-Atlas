"""
Atlas TNT Configuration Schema - Pydantic V2

This is the SINGLE SOURCE OF TRUTH for all configuration.

Rules:
1. NO DEFAULT VALUES unless explicitly documented why
2. If a field is not in the config YAML, the script FAILS
3. All types are validated at load time
4. Use Optional[] only when None is a valid runtime value

Usage:
    from config_schema import load_config, AtlasConfig

    config = load_config("configs/default.yaml")  # Fails fast if invalid
    print(config.model.d_model)  # Type-safe access

To add a new config field:
1. Add it to the appropriate Pydantic model below
2. Add it to configs/default.yaml
3. The script will fail until both are done
"""

from pathlib import Path
from typing import Optional, List, Literal, Any, Dict, Type
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml


# ═══════════════════════════════════════════════════════════════════════════════
# Model Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class ModelConfig(BaseModel):
    """Neural network architecture configuration."""

    # Vocabulary - MUST match tokenizer
    vocab_size: int = Field(..., description="Vocabulary size, must match tokenizer")

    # Core dimensions
    d_model: int = Field(..., description="Model embedding dimension")
    n_heads: int = Field(..., description="Number of attention heads")
    n_layers: int = Field(..., description="Number of AtlasMAG blocks")

    # MAG (Memory-Augmented Gating) parameters
    window_size: int = Field(..., description="Sliding window attention size")
    n_persistent: int = Field(..., description="Number of persistent memory tokens")

    # Atlas memory parameters
    omega_context: int = Field(..., description="Omega rule context window size")
    poly_degree: int = Field(..., description="Polynomial feature degree (2 = O(d²) capacity)")
    ns_iterations: int = Field(..., description="Base Newton-Schulz iterations")

    # M3 Mixing parameters
    m3_alpha_target: float = Field(..., description="Target alpha after warmup")
    m3_alpha_start: float = Field(..., description="Initial alpha during warmup")
    m3_warmup_steps: int = Field(..., description="Steps to ramp alpha from start to target")

    # Safety parameters
    qk_gain_circuit_breaker: float = Field(..., description="Max QK gain before freezing")
    saturation_kill_threshold: float = Field(..., description="Saturation % that triggers kill switch")
    saturation_kill_patience: int = Field(..., description="Steps of saturation before kill")

    # Regularization
    dropout: float = Field(..., description="Dropout probability")

    # Derived/hardcoded values that SHOULD be configurable (moved from hardcoded)
    eta_scale: float = Field(..., description="Learning rate gate scaling factor")

    @field_validator('d_model')
    @classmethod
    def d_model_divisible_by_heads(cls, v: int, info: Any) -> int:
        # Can't validate against n_heads here, do it in model_validator
        return v

    @model_validator(mode='after')
    def validate_dimensions(self) -> "ModelConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.poly_degree < 1 or self.poly_degree > 3:
            raise ValueError(f"poly_degree must be 1, 2, or 3, got {self.poly_degree}")
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class Stage1Config(BaseModel):
    """TNT Stage 1 (Efficiency-focused) configuration."""

    shard_length: int = Field(..., description="Token count per shard (reset interval)")
    total_tokens: int = Field(..., description="Total tokens to train on")
    chunk_size: int = Field(..., description="Tokens per training chunk")

    @model_validator(mode='after')
    def validate_shard_chunk_relationship(self) -> "Stage1Config":
        if self.shard_length % self.chunk_size != 0:
            raise ValueError(
                f"shard_length ({self.shard_length}) must be divisible by "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class Stage2Config(BaseModel):
    """TNT Stage 2 (Performance-focused) configuration."""

    chunk_sizes: List[int] = Field(..., description="Multi-resolution chunk sizes")
    tokens_per_chunk: int = Field(..., description="Tokens to train per chunk size")


class LRConfig(BaseModel):
    """Learning rate schedule configuration."""

    peak: float = Field(..., description="Peak learning rate")
    min: float = Field(..., description="Minimum learning rate")
    warmup_steps: int = Field(..., description="Warmup steps")
    schedule: Literal["cosine", "linear", "constant"] = Field(..., description="LR schedule type")


class TrainingConfig(BaseModel):
    """Training loop configuration."""

    # TNT stages
    stage1: Stage1Config
    stage2: Stage2Config

    # Optimizer
    optimizer: Literal["AdamW", "Adam", "SGD"] = Field(..., description="Optimizer type")
    weight_decay: float = Field(..., description="Weight decay for AdamW")
    grad_clip: float = Field(..., description="Gradient clipping norm")

    # Learning rate
    lr: LRConfig

    # Differential LR (neocortex vs hippocampus)
    differential_lr_ratio: float = Field(..., description="Global LR = base_lr * this ratio")

    # Batch configuration
    batch_size: int = Field(..., description="Batch size per step")
    gradient_accumulation: int = Field(..., description="Gradient accumulation steps")

    # Checkpointing
    checkpoint_interval: int = Field(..., description="Steps between checkpoints")

    @model_validator(mode='after')
    def validate_lr(self) -> "TrainingConfig":
        if self.lr.min > self.lr.peak:
            raise ValueError(f"lr.min ({self.lr.min}) cannot be greater than lr.peak ({self.lr.peak})")
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class CacheConfig(BaseModel):
    """Hybrid caching configuration."""

    enabled: bool = Field(..., description="Whether caching is enabled")
    lazy_threshold: int = Field(..., description="0 = strict, >0 = allow micro-staleness")
    gradient_magnitude_threshold: float = Field(..., description="Weight drift invalidation threshold")


# ═══════════════════════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class TelemetryConfig(BaseModel):
    """Telemetry and monitoring configuration."""

    log_interval: int = Field(..., description="Steps between log outputs")
    track_saturation: bool = Field(..., description="Track tanh saturation")
    track_alpha: bool = Field(..., description="Track M3 alpha values")
    track_cache_stats: bool = Field(..., description="Track cache hit/miss stats")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(..., description="Log level")
    tensorboard: bool = Field(..., description="Enable TensorBoard logging")
    wandb: bool = Field(..., description="Enable W&B logging")
    telemetry: TelemetryConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class HardwareConfig(BaseModel):
    """Hardware and device configuration."""

    device: Literal["cuda", "cpu", "mps"] = Field(..., description="Compute device")
    dtype: Literal["float32", "float16", "bfloat16"] = Field(..., description="Data type")
    distributed: bool = Field(..., description="Enable distributed training")


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Gates Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationGatesConfig(BaseModel):
    """Pre-training validation gates."""

    isolation_suite: bool = Field(..., description="Run isolation suite before training")
    max_nan_steps: int = Field(..., description="Max NaN steps before failure (0 = immediate)")
    max_loss_spike_ratio: float = Field(..., description="Max loss spike ratio before warning")
    min_alpha_range: List[float] = Field(..., description="[min, max] alpha range to avoid collapse")

    @field_validator('min_alpha_range')
    @classmethod
    def validate_alpha_range(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("min_alpha_range must be [min, max]")
        if v[0] >= v[1]:
            raise ValueError(f"min_alpha_range[0] ({v[0]}) must be < min_alpha_range[1] ({v[1]})")
        return v


# ═══════════════════════════════════════════════════════════════════════════════
# Paths Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class PathsConfig(BaseModel):
    """File and directory paths."""

    data_dir: str = Field(..., description="Data directory")
    checkpoint_dir: str = Field(..., description="Checkpoint output directory")
    log_dir: str = Field(..., description="Log output directory")
    tensorboard_dir: str = Field(..., description="TensorBoard output directory")


# ═══════════════════════════════════════════════════════════════════════════════
# Data Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class DataConfig(BaseModel):
    """Dataset and tokenizer configuration."""

    # HuggingFace dataset (for streaming)
    dataset: str = Field(..., description="HuggingFace dataset name")
    subset: Optional[str] = Field(..., description="Dataset subset/config (None if not needed)")
    text_column: str = Field(..., description="Column name containing text")

    # Tokenizer
    tokenizer: str = Field(..., description="Tokenizer name (must match model.vocab_size)")

    # Pre-tokenized data path (None = use streaming)
    pretokenized_path: Optional[str] = Field(..., description="Path to pre-tokenized data dir (None = stream)")

    # Streaming mode (ignored if pretokenized_path is set)
    streaming: bool = Field(..., description="Stream from HuggingFace (vs download)")


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Run Configuration (separate from training)
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationRunConfig(BaseModel):
    """Configuration for running validation separately from training."""

    gpu: int = Field(..., description="GPU ID for validation (separate from training GPU)")
    max_batches: Optional[int] = Field(..., description="Max batches per validation (None = all)")
    watch_interval: int = Field(..., description="Seconds between checkpoint watches")


# ═══════════════════════════════════════════════════════════════════════════════
# M3 Safety Configuration (previously hardcoded)
# ═══════════════════════════════════════════════════════════════════════════════

class M3SafetyConfig(BaseModel):
    """M3 mixing safety thresholds - previously hardcoded, now configurable."""

    danger_high: float = Field(..., description="Alpha above this triggers warning")
    danger_low: float = Field(..., description="Alpha below this triggers warning")
    post_warmup_monitor_steps: int = Field(..., description="Steps to monitor after warmup")
    regularization_low_boundary: float = Field(..., description="Lower boundary for regularization loss")
    regularization_high_boundary: float = Field(..., description="Upper boundary for regularization loss")


# ═══════════════════════════════════════════════════════════════════════════════
# Newton-Schulz Configuration (previously hardcoded)
# ═══════════════════════════════════════════════════════════════════════════════

class NewtonSchulzConfig(BaseModel):
    """Newton-Schulz iteration configuration - previously hardcoded."""

    warmup_steps: int = Field(..., description="Warmup period for K decay")
    k_max: int = Field(..., description="Max iterations at shard boundary")
    k_min: int = Field(..., description="Min iterations after warmup")


# ═══════════════════════════════════════════════════════════════════════════════
# Seed Model Configuration (Ad Hoc Level Stacking - NL Paper Section 7.3)
# ═══════════════════════════════════════════════════════════════════════════════

class SeedModelConfig(BaseModel):
    """
    Seed model configuration for ad hoc level stacking.

    From Nested Learning paper Section 7.3: The seed model provides initial
    weights for the CMS/memory blocks, solving the "cold start" problem.
    Without this, random initialization guarantees instability pockets.
    """

    enabled: bool = Field(..., description="Whether to use seed model for initialization")
    model_name: str = Field(..., description="HuggingFace model ID for seed model")
    cache_dir: Optional[str] = Field(..., description="Local cache directory (None = default)")

    # Source layer selection
    source_layers: List[int] = Field(
        ...,
        description="Which layers to extract from seed model (0-indexed)"
    )

    # Weight mapping strategy
    mapping_strategy: Literal["mlp_to_memory", "attention_to_qk"] = Field(
        ...,
        description="How to map seed weights to Atlas blocks"
    )

    # Learning rate multiplier for seeded weights
    seed_lr_multiplier: float = Field(
        ...,
        description="LR multiplier for seeded weights (lower = stay closer to seed)"
    )

    @field_validator('source_layers')
    @classmethod
    def validate_source_layers(cls, v: List[int]) -> List[int]:
        if len(v) == 0:
            raise ValueError("source_layers cannot be empty")
        if any(layer < 0 for layer in v):
            raise ValueError("source_layers must be non-negative")
        return v

    @field_validator('seed_lr_multiplier')
    @classmethod
    def validate_lr_multiplier(cls, v: float) -> float:
        if v <= 0 or v > 1:
            raise ValueError(f"seed_lr_multiplier must be in (0, 1], got {v}")
        return v


# ═══════════════════════════════════════════════════════════════════════════════
# Root Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class AtlasConfig(BaseModel):
    """
    Root configuration for Atlas TNT Training.

    ALL fields are REQUIRED. No defaults.
    If your YAML is missing a field, you'll get a clear error.
    """

    model: ModelConfig
    training: TrainingConfig
    cache: CacheConfig
    logging: LoggingConfig
    hardware: HardwareConfig
    validation: ValidationGatesConfig
    paths: PathsConfig
    data: DataConfig
    validation_run: ValidationRunConfig
    m3_safety: M3SafetyConfig
    newton_schulz: NewtonSchulzConfig
    seed_model: SeedModelConfig

    class Config:
        # Forbid extra fields - catches typos in YAML
        extra = 'forbid'


# ═══════════════════════════════════════════════════════════════════════════════
# Config Loading Functions
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str | Path) -> AtlasConfig:
    """
    Load and validate configuration from YAML file.

    Fails fast with clear error message if:
    - File doesn't exist
    - YAML is malformed
    - Any required field is missing
    - Any field has wrong type
    - Any field fails validation

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated AtlasConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
        pydantic.ValidationError: If config is invalid (with detailed error message)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Pydantic V2 validation - fails fast with clear errors
    return AtlasConfig.model_validate(raw_config)


def load_config_partial(config_path: str | Path, required_sections: List[str]) -> dict:
    """
    Load config with validation only for specified sections.

    Useful for scripts that don't need the full config (e.g., prepare_data.py
    doesn't need model config).

    Args:
        config_path: Path to YAML config file
        required_sections: List of section names that must be present and valid

    Returns:
        Raw dict for non-required sections, validated models for required sections
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Validate required sections
    section_models: Dict[str, Type[BaseModel]] = {
        'model': ModelConfig,
        'training': TrainingConfig,
        'cache': CacheConfig,
        'logging': LoggingConfig,
        'hardware': HardwareConfig,
        'validation': ValidationGatesConfig,
        'paths': PathsConfig,
        'data': DataConfig,
        'validation_run': ValidationRunConfig,
        'm3_safety': M3SafetyConfig,
        'newton_schulz': NewtonSchulzConfig,
        'seed_model': SeedModelConfig,
    }

    result: Dict[str, Any] = {}
    for section in required_sections:
        if section not in raw_config:
            raise ValueError(f"Required config section missing: {section}")
        if section in section_models:
            result[section] = section_models[section].model_validate(raw_config[section])
        else:
            result[section] = raw_config[section]

    # Include non-required sections as raw dicts
    for key, value in raw_config.items():
        if key not in result:
            result[key] = value

    return result
