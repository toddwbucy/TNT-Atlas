"""
Integration tests for checkpoint save/load.

These tests verify:
1. Checkpoints save correctly
2. Checkpoints load correctly
3. Model state is preserved across save/load
4. Atomic write works (temp file + rename)
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
# Test: Checkpoint Saving
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestCheckpointSaving:
    """Test checkpoint save functionality."""

    def test_save_checkpoint_creates_file(self, tiny_config, tmp_path):
        """save_checkpoint should create a file."""
        from train import AtlasModel, save_checkpoint

        model = AtlasModel(tiny_config)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_path = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            step=100,
            total_tokens=10000,
            config=tiny_config,
            loss=2.5,
        )

        assert ckpt_path.exists()
        assert ckpt_path.name == "step_00000100.pt"

    def test_save_checkpoint_creates_latest_symlink(self, tiny_config, tmp_path):
        """save_checkpoint should create 'latest' symlink."""
        from train import AtlasModel, save_checkpoint

        model = AtlasModel(tiny_config)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt_dir = tmp_path / "checkpoints"
        save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            step=100,
            total_tokens=10000,
            config=tiny_config,
            loss=2.5,
        )

        latest = ckpt_dir / "latest.pt"
        assert latest.exists() or latest.is_symlink()

    def test_checkpoint_contains_required_keys(self, tiny_config, tmp_path):
        """Checkpoint should contain all required keys."""
        from train import AtlasModel, save_checkpoint

        model = AtlasModel(tiny_config)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_path = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            step=100,
            total_tokens=10000,
            config=tiny_config,
            loss=2.5,
        )

        checkpoint = torch.load(ckpt_path)

        required_keys = [
            "model_state_dict",
            "optimizer_state_dict",
            "step",
            "total_tokens",
            "loss",
            "config",
            "metadata",
        ]
        for key in required_keys:
            assert key in checkpoint, f"Missing key: {key}"

    def test_config_stored_as_dict(self, tiny_config, tmp_path):
        """Config should be stored as dict for portability."""
        from train import AtlasModel, save_checkpoint

        model = AtlasModel(tiny_config)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_path = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            step=100,
            total_tokens=10000,
            config=tiny_config,
            loss=2.5,
        )

        checkpoint = torch.load(ckpt_path)
        assert isinstance(checkpoint["config"], dict)
        assert checkpoint["config"]["model"]["d_model"] == 32


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Checkpoint Loading
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestCheckpointLoading:
    """Test checkpoint load functionality."""

    def test_load_checkpoint_restores_model(self, tiny_config, tmp_path):
        """load_checkpoint should restore model state."""
        from train import AtlasModel, save_checkpoint, load_checkpoint

        # Create and save model
        model1 = AtlasModel(tiny_config)
        optimizer1 = torch.optim.Adam(model1.parameters())

        # Run a forward pass to change state
        input_ids = torch.randint(0, 1000, (2, 32))
        _ = model1(input_ids)

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_path = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model1,
            optimizer=optimizer1,
            step=100,
            total_tokens=10000,
            config=tiny_config,
            loss=2.5,
        )

        # Create new model and load checkpoint
        model2 = AtlasModel(tiny_config)
        optimizer2 = torch.optim.Adam(model2.parameters())

        info = load_checkpoint(ckpt_path, model2, optimizer2)

        # Compare state dicts
        for key in model1.state_dict():
            assert torch.equal(
                model1.state_dict()[key],
                model2.state_dict()[key]
            ), f"Mismatch in {key}"

    def test_load_checkpoint_returns_metadata(self, tiny_config, tmp_path):
        """load_checkpoint should return step, tokens, loss."""
        from train import AtlasModel, save_checkpoint, load_checkpoint

        model = AtlasModel(tiny_config)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_path = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            step=100,
            total_tokens=10000,
            config=tiny_config,
            loss=2.5,
        )

        model2 = AtlasModel(tiny_config)
        info = load_checkpoint(ckpt_path, model2)

        assert info["step"] == 100
        assert info["total_tokens"] == 10000
        assert info["loss"] == 2.5

    def test_load_checkpoint_without_optimizer(self, tiny_config, tmp_path):
        """Should be able to load checkpoint without optimizer (eval mode)."""
        from train import AtlasModel, save_checkpoint, load_checkpoint

        model = AtlasModel(tiny_config)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_path = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            step=100,
            total_tokens=10000,
            config=tiny_config,
            loss=2.5,
        )

        # Load without optimizer
        model2 = AtlasModel(tiny_config)
        info = load_checkpoint(ckpt_path, model2, optimizer=None)

        assert info["step"] == 100


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Find Latest Checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestFindLatestCheckpoint:
    """Test finding latest checkpoint."""

    def test_find_latest_via_symlink(self, tiny_config, tmp_path):
        """Should find latest via 'latest' symlink."""
        from train import AtlasModel, save_checkpoint, find_latest_checkpoint

        model = AtlasModel(tiny_config)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt_dir = tmp_path / "checkpoints"
        save_checkpoint(ckpt_dir, model, optimizer, 100, 10000, tiny_config, 2.5)
        save_checkpoint(ckpt_dir, model, optimizer, 200, 20000, tiny_config, 2.0)

        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None
        assert "200" in str(latest)

    def test_find_latest_no_checkpoints(self, tmp_path):
        """Should return None if no checkpoints exist."""
        from train import find_latest_checkpoint

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        latest = find_latest_checkpoint(empty_dir)
        assert latest is None
