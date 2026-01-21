"""
Unit tests for config_schema.py - Pydantic V2 configuration validation.

These tests verify:
1. Required fields raise errors when missing
2. Type validation works correctly
3. Cross-field validation (e.g., d_model % n_heads == 0)
4. No hidden defaults exist
5. Config loading from YAML works
"""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from src.config_schema import (
    AtlasConfig,
    ModelConfig,
    TrainingConfig,
    CacheConfig,
    LoggingConfig,
    HardwareConfig,
    ValidationGatesConfig,
    PathsConfig,
    DataConfig,
    ValidationRunConfig,
    M3SafetyConfig,
    NewtonSchulzConfig,
    load_config,
    load_config_partial,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Required Fields (No Hidden Defaults)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRequiredFields:
    """Verify all fields are required (no hidden defaults)."""

    def test_model_config_missing_field_fails(self):
        """Missing vocab_size should fail."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                # vocab_size missing
                d_model=512,
                n_heads=8,
                n_layers=6,
                window_size=256,
                n_persistent=4,
                omega_context=8,
                poly_degree=2,
                ns_iterations=1,
                m3_alpha_target=0.5,
                m3_alpha_start=0.1,
                m3_warmup_steps=500,
                qk_gain_circuit_breaker=2.0,
                saturation_kill_threshold=0.20,
                saturation_kill_patience=100,
                dropout=0.1,
                eta_scale=0.1,
            )
        assert "vocab_size" in str(exc_info.value).lower()

    def test_training_config_missing_nested_field_fails(self):
        """Missing nested field in stage1 should fail."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                stage1={
                    "shard_length": 2048,
                    # total_tokens missing
                    "chunk_size": 128,
                },
                stage2={"chunk_sizes": [2, 4], "tokens_per_chunk": 10000000},
                optimizer="AdamW",
                weight_decay=0.1,
                grad_clip=1.0,
                lr={"peak": 0.0001, "min": 0.000001, "warmup_steps": 1000, "schedule": "cosine"},
                differential_lr_ratio=0.01,
                batch_size=4,
                gradient_accumulation=8,
                checkpoint_interval=1000,
            )

    def test_cache_config_requires_all_fields(self):
        """Cache config requires all three fields."""
        with pytest.raises(ValidationError):
            CacheConfig(
                enabled=True,
                lazy_threshold=0,
                # gradient_magnitude_threshold missing
            )

    def test_hardware_config_requires_all_fields(self):
        """Hardware config requires device, dtype, distributed."""
        with pytest.raises(ValidationError):
            HardwareConfig(
                device="cuda",
                # dtype missing
                distributed=False,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Type Validation
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTypeValidation:
    """Verify type coercion and validation."""

    def test_string_to_int_coercion(self, tiny_config_dict):
        """Pydantic should coerce "512" to 512."""
        config_dict = tiny_config_dict.copy()
        config_dict["model"]["d_model"] = "32"  # String instead of int

        # Should succeed with coercion
        config = AtlasConfig.model_validate(config_dict)
        assert config.model.d_model == 32
        assert isinstance(config.model.d_model, int)

    def test_invalid_literal_fails(self, tiny_config_dict):
        """Invalid literal value should fail."""
        config_dict = tiny_config_dict.copy()
        config_dict["hardware"]["device"] = "tpu"  # Not in Literal["cuda", "cpu", "mps"]

        with pytest.raises(ValidationError) as exc_info:
            AtlasConfig.model_validate(config_dict)
        assert "device" in str(exc_info.value).lower()

    def test_invalid_schedule_fails(self, tiny_config_dict):
        """Invalid LR schedule should fail."""
        config_dict = tiny_config_dict.copy()
        config_dict["training"]["lr"]["schedule"] = "exponential"  # Not valid

        with pytest.raises(ValidationError):
            AtlasConfig.model_validate(config_dict)

    def test_float_validation(self, tiny_config_dict):
        """Float fields should accept int and coerce."""
        config_dict = tiny_config_dict.copy()
        config_dict["training"]["weight_decay"] = 1  # Int instead of float

        config = AtlasConfig.model_validate(config_dict)
        assert config.training.weight_decay == 1.0
        assert isinstance(config.training.weight_decay, float)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Cross-Field Validation
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCrossFieldValidation:
    """Verify cross-field validation rules."""

    def test_d_model_divisible_by_n_heads(self, tiny_config_dict):
        """d_model must be divisible by n_heads."""
        config_dict = tiny_config_dict.copy()
        config_dict["model"]["d_model"] = 33  # Not divisible by 4
        config_dict["model"]["n_heads"] = 4

        with pytest.raises(ValidationError) as exc_info:
            AtlasConfig.model_validate(config_dict)
        assert "divisible" in str(exc_info.value).lower()

    def test_shard_length_divisible_by_chunk_size(self, tiny_config_dict):
        """shard_length must be divisible by chunk_size."""
        config_dict = tiny_config_dict.copy()
        config_dict["training"]["stage1"]["shard_length"] = 100
        config_dict["training"]["stage1"]["chunk_size"] = 33  # 100 % 33 != 0

        with pytest.raises(ValidationError) as exc_info:
            AtlasConfig.model_validate(config_dict)
        assert "divisible" in str(exc_info.value).lower()

    def test_lr_min_cannot_exceed_peak(self, tiny_config_dict):
        """lr.min cannot be greater than lr.peak."""
        config_dict = tiny_config_dict.copy()
        config_dict["training"]["lr"]["min"] = 0.01
        config_dict["training"]["lr"]["peak"] = 0.001  # min > peak

        with pytest.raises(ValidationError) as exc_info:
            AtlasConfig.model_validate(config_dict)
        assert "min" in str(exc_info.value).lower() or "peak" in str(exc_info.value).lower()

    def test_poly_degree_valid_range(self, tiny_config_dict):
        """poly_degree must be 1, 2, or 3."""
        config_dict = tiny_config_dict.copy()
        config_dict["model"]["poly_degree"] = 5  # Out of range

        with pytest.raises(ValidationError) as exc_info:
            AtlasConfig.model_validate(config_dict)
        assert "poly_degree" in str(exc_info.value).lower()

    def test_alpha_range_validation(self, tiny_config_dict):
        """min_alpha_range must have [min, max] where min < max."""
        config_dict = tiny_config_dict.copy()
        config_dict["validation"]["min_alpha_range"] = [0.9, 0.1]  # Reversed

        with pytest.raises(ValidationError):
            AtlasConfig.model_validate(config_dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Config Loading
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestConfigLoading:
    """Test loading config from YAML files."""

    def test_load_valid_config(self, tiny_config_path):
        """Loading valid config should succeed."""
        config = load_config(tiny_config_path)
        assert isinstance(config, AtlasConfig)
        assert config.model.d_model == 32

    def test_load_missing_file_fails(self, tmp_path):
        """Loading non-existent file should fail."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_empty_file_fails(self, tmp_path):
        """Loading empty YAML file should fail."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ValueError) as exc_info:
            load_config(empty_file)
        assert "empty" in str(exc_info.value).lower()

    def test_load_malformed_yaml_fails(self, tmp_path):
        """Loading malformed YAML should fail."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("this: is: bad: yaml:")

        with pytest.raises(Exception):  # yaml.YAMLError
            load_config(bad_file)

    def test_extra_fields_forbidden(self, tiny_config_dict, tmp_path):
        """Extra fields in config should fail (catches typos)."""
        config_dict = tiny_config_dict.copy()
        config_dict["typo_field"] = "oops"

        config_file = tmp_path / "typo.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)

        with pytest.raises(ValidationError) as exc_info:
            load_config(config_file)
        assert "extra" in str(exc_info.value).lower() or "typo" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Attribute Access
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestAttributeAccess:
    """Test that config values are accessible via attributes."""

    def test_nested_attribute_access(self, tiny_config):
        """Should access nested config via attributes."""
        assert tiny_config.model.d_model == 32
        assert tiny_config.training.lr.peak == 0.001
        assert tiny_config.hardware.device == "cpu"

    def test_model_dump_to_dict(self, tiny_config):
        """model_dump() should convert to dict."""
        config_dict = tiny_config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["model"]["d_model"] == 32

    def test_attribute_types(self, tiny_config):
        """Verify attribute types are correct."""
        assert isinstance(tiny_config.model.d_model, int)
        assert isinstance(tiny_config.training.weight_decay, float)
        assert isinstance(tiny_config.cache.enabled, bool)
        assert isinstance(tiny_config.training.stage1.chunk_sizes if hasattr(tiny_config.training.stage1, 'chunk_sizes') else tiny_config.training.stage2.chunk_sizes, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Partial Loading
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPartialLoading:
    """Test load_config_partial for scripts that don't need full config."""

    def test_partial_load_data_section(self, tiny_config_path):
        """Should load and validate only specified sections."""
        result = load_config_partial(tiny_config_path, required_sections=["data", "paths"])

        # Required sections should be validated models
        assert hasattr(result["data"], "dataset")
        assert hasattr(result["paths"], "data_dir")

        # Other sections should be raw dicts
        assert isinstance(result["model"], dict)

    def test_partial_load_missing_section_fails(self, tmp_path):
        """Should fail if required section is missing."""
        incomplete = tmp_path / "incomplete.yaml"
        with open(incomplete, 'w') as f:
            yaml.dump({"model": {"d_model": 32}}, f)

        with pytest.raises(ValueError) as exc_info:
            load_config_partial(incomplete, required_sections=["data"])
        assert "data" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Real Config File
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRealConfig:
    """Test loading the actual default.yaml config."""

    def test_load_default_config(self):
        """default.yaml should load without errors."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        if config_path.exists():
            config = load_config(config_path)
            # vocab_size=49152 to match SmolLM-135M seed model
            assert config.model.vocab_size == 49152
            assert config.training.stage1.total_tokens == 100_000_000  # Quick validation run
            # Verify seed model config loaded
            assert config.seed_model.enabled is True
            assert config.seed_model.model_name == "HuggingFaceTB/SmolLM-135M"
        else:
            pytest.skip("default.yaml not found")

    def test_default_config_no_hidden_defaults(self):
        """Verify default.yaml has all required fields."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        if not config_path.exists():
            pytest.skip("default.yaml not found")

        # This should not raise - all fields present
        config = load_config(config_path)

        # Spot check some fields that were previously hardcoded
        assert hasattr(config.model, "eta_scale")
        assert hasattr(config.m3_safety, "danger_high")
        assert hasattr(config.newton_schulz, "k_max")
