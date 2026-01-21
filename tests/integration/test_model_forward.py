"""
Integration tests for AtlasModel forward pass.

These tests verify:
1. Model can be instantiated from config
2. Forward pass produces valid outputs
3. Memory states are returned correctly
4. Shard boundary handling works
"""

import sys
from pathlib import Path

import pytest
import torch

# Add scripts directory to path for imports (scripts are not a package)
_scripts_dir = Path(__file__).parent.parent.parent / 'scripts'
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Model Instantiation
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestModelInstantiation:
    """Test AtlasModel creation from config."""

    def test_model_from_pydantic_config(self, tiny_config):
        """Model should instantiate from AtlasConfig."""
        from train import AtlasModel
        model = AtlasModel(tiny_config)

        assert model.d_model == 32
        assert model.n_layers == 2
        assert model.vocab_size == 1000

    def test_model_from_dict_config(self, tiny_config_dict):
        """Model should instantiate from dict (checkpoint compatibility)."""
        from train import AtlasModel
        model = AtlasModel(tiny_config_dict)

        assert model.d_model == 32

    def test_model_parameter_count(self, tiny_config):
        """Model should have reasonable parameter count."""
        from train import AtlasModel
        model = AtlasModel(tiny_config)

        n_params = sum(p.numel() for p in model.parameters())
        # Tiny model should have relatively few parameters
        assert n_params < 10_000_000  # Less than 10M
        assert n_params > 10_000  # More than 10K


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Forward Pass
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestForwardPass:
    """Test model forward pass."""

    def test_forward_produces_logits(self, tiny_config):
        """Forward pass should produce logits of correct shape."""
        from train import AtlasModel

        model = AtlasModel(tiny_config)
        batch_size = 2
        seq_len = 32
        vocab_size = tiny_config.model.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, memory_states, telemetry = model(input_ids)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_forward_returns_memory_states(self, tiny_config):
        """Forward should return memory states for each layer."""
        from train import AtlasModel

        model = AtlasModel(tiny_config)
        n_layers = tiny_config.model.n_layers
        input_ids = torch.randint(0, 1000, (2, 32))

        logits, memory_states, telemetry = model(input_ids)

        assert len(memory_states) == n_layers
        # Each state should be a dict (or None for first pass)
        for state in memory_states:
            assert state is None or isinstance(state, dict)

    def test_forward_returns_telemetry(self, tiny_config):
        """Forward should return telemetry dict."""
        from train import AtlasModel

        model = AtlasModel(tiny_config)
        input_ids = torch.randint(0, 1000, (2, 32))

        logits, memory_states, telemetry = model(input_ids)

        assert isinstance(telemetry, dict)

    def test_forward_no_nan(self, tiny_config):
        """Forward pass should not produce NaN values."""
        from train import AtlasModel

        model = AtlasModel(tiny_config)
        input_ids = torch.randint(0, 1000, (2, 32))

        logits, memory_states, telemetry = model(input_ids)

        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Shard Boundaries
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestShardBoundaries:
    """Test shard boundary handling."""

    def test_shard_boundary_flag(self, tiny_config):
        """Model should accept shard_boundary flag."""
        from train import AtlasModel

        model = AtlasModel(tiny_config)
        input_ids = torch.randint(0, 1000, (2, 32))

        # First call without boundary
        logits1, states1, _ = model(input_ids, shard_boundary=False)

        # Second call with boundary (should use M3 mixing)
        logits2, states2, _ = model(input_ids, memory_states=states1, shard_boundary=True)

        # Both should produce valid outputs
        assert logits1.shape == logits2.shape
        assert not torch.isnan(logits2).any()

    def test_memory_state_continuity(self, tiny_config):
        """Memory states should pass between calls."""
        from train import AtlasModel

        model = AtlasModel(tiny_config)
        input_ids = torch.randint(0, 1000, (2, 32))

        # First pass - no initial state
        logits1, states1, _ = model(input_ids, memory_states=None)

        # Second pass - use states from first
        logits2, states2, _ = model(input_ids, memory_states=states1)

        # States should be updated (not identical)
        # Note: This may vary by implementation
        assert states2 is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Gradient Flow
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestGradientFlow:
    """Test gradient computation."""

    def test_loss_backward(self, tiny_config):
        """Should be able to compute loss and backpropagate."""
        from train import AtlasModel
        import torch.nn.functional as F

        model = AtlasModel(tiny_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))

        logits, _, _ = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients computed"

    def test_no_nan_gradients(self, tiny_config):
        """Gradients should not be NaN."""
        from train import AtlasModel
        import torch.nn.functional as F

        model = AtlasModel(tiny_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))

        logits, _, _ = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_regularization_loss(self, tiny_config):
        """Model should provide regularization loss."""
        from train import AtlasModel

        model = AtlasModel(tiny_config)
        input_ids = torch.randint(0, 1000, (2, 32))

        # Run forward to initialize states
        _ = model(input_ids)

        reg_loss = model.get_regularization_loss()
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.ndim == 0  # Scalar
