"""
Unit tests for data_loader.py - TNT-aware data loading.

These tests verify:
1. PreTokenizedDataset correctly chunks and tracks shard boundaries
2. Collation function works correctly
3. Chinchilla calculations are correct
4. No hidden dependencies on network/HuggingFace
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.data_loader import (
    PreTokenizedDataset,
    tnt_collate_fn,
    chinchilla_optimal_tokens,
    estimate_training_steps,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: PreTokenizedDataset
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPreTokenizedDataset:
    """Test PreTokenizedDataset chunking and shard boundary tracking."""

    def test_dataset_length(self, tiny_tokens_path):
        """Dataset length should be (n_tokens - 1) // chunk_size."""
        dataset = PreTokenizedDataset(
            data_path=str(tiny_tokens_path),
            chunk_size=32,
            shard_length=128,
        )
        # 10000 tokens, chunk_size=32 -> (10000-1)//32 = 312 chunks
        expected_length = (10000 - 1) // 32
        assert len(dataset) == expected_length

    def test_chunk_shapes(self, tiny_tokens_path):
        """Each chunk should have correct shapes."""
        chunk_size = 32
        dataset = PreTokenizedDataset(
            data_path=str(tiny_tokens_path),
            chunk_size=chunk_size,
            shard_length=128,
        )

        item = dataset[0]
        assert item["input_ids"].shape == (chunk_size,)
        assert item["labels"].shape == (chunk_size,)

    def test_input_label_offset(self, tiny_tokens_path):
        """Labels should be input_ids shifted by 1."""
        dataset = PreTokenizedDataset(
            data_path=str(tiny_tokens_path),
            chunk_size=32,
            shard_length=128,
        )

        item = dataset[0]
        # Load raw tokens to verify
        raw_tokens = np.load(tiny_tokens_path)

        # input_ids should be tokens[0:32]
        # labels should be tokens[1:33]
        assert torch.equal(item["input_ids"], torch.tensor(raw_tokens[0:32], dtype=torch.long))
        assert torch.equal(item["labels"], torch.tensor(raw_tokens[1:33], dtype=torch.long))

    def test_shard_boundary_first_chunk(self, tiny_tokens_path):
        """First chunk should NOT be a shard boundary."""
        dataset = PreTokenizedDataset(
            data_path=str(tiny_tokens_path),
            chunk_size=32,
            shard_length=128,
        )

        item = dataset[0]
        assert item["shard_boundary"] == False

    def test_shard_boundary_at_shard_start(self, tiny_tokens_path):
        """Chunk at shard boundary should have shard_boundary=True."""
        chunk_size = 32
        shard_length = 128  # 128/32 = 4 chunks per shard

        dataset = PreTokenizedDataset(
            data_path=str(tiny_tokens_path),
            chunk_size=chunk_size,
            shard_length=shard_length,
        )

        # Chunk 0: position 0, not boundary
        # Chunk 1: position 32, not boundary
        # Chunk 2: position 64, not boundary
        # Chunk 3: position 96, not boundary
        # Chunk 4: position 128 = shard_length, IS boundary

        assert dataset[0]["shard_boundary"] == False
        assert dataset[1]["shard_boundary"] == False
        assert dataset[2]["shard_boundary"] == False
        assert dataset[3]["shard_boundary"] == False
        assert dataset[4]["shard_boundary"] == True  # New shard
        assert dataset[5]["shard_boundary"] == False
        assert dataset[8]["shard_boundary"] == True  # Another shard boundary

    def test_different_chunk_sizes(self, tiny_tokens_path):
        """Different chunk sizes should produce correct shapes."""
        for chunk_size in [16, 64, 128]:
            dataset = PreTokenizedDataset(
                data_path=str(tiny_tokens_path),
                chunk_size=chunk_size,
                shard_length=256,
            )
            item = dataset[0]
            assert item["input_ids"].shape == (chunk_size,)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Collation
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCollation:
    """Test TNT collation function."""

    def test_collate_stacks_tensors(self):
        """Collation should stack input_ids and labels."""
        batch = [
            {"input_ids": torch.arange(32), "labels": torch.arange(1, 33), "shard_boundary": False},
            {"input_ids": torch.arange(32), "labels": torch.arange(1, 33), "shard_boundary": False},
        ]

        result = tnt_collate_fn(batch)
        assert result["input_ids"].shape == (2, 32)
        assert result["labels"].shape == (2, 32)

    def test_collate_shard_boundary_any(self):
        """shard_boundary should be True if ANY item has boundary."""
        batch_no_boundary = [
            {"input_ids": torch.zeros(32), "labels": torch.zeros(32), "shard_boundary": False},
            {"input_ids": torch.zeros(32), "labels": torch.zeros(32), "shard_boundary": False},
        ]
        assert tnt_collate_fn(batch_no_boundary)["shard_boundary"] == False

        batch_with_boundary = [
            {"input_ids": torch.zeros(32), "labels": torch.zeros(32), "shard_boundary": False},
            {"input_ids": torch.zeros(32), "labels": torch.zeros(32), "shard_boundary": True},
        ]
        assert tnt_collate_fn(batch_with_boundary)["shard_boundary"] == True

    def test_collate_single_item(self):
        """Collation should work with batch size 1."""
        batch = [
            {"input_ids": torch.arange(32), "labels": torch.arange(1, 33), "shard_boundary": True},
        ]

        result = tnt_collate_fn(batch)
        assert result["input_ids"].shape == (1, 32)
        assert result["shard_boundary"] == True


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Chinchilla Calculations
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestChinchillaCalculations:
    """Test Chinchilla optimal scaling calculations."""

    def test_chinchilla_50m_params(self):
        """50M params should give 1B tokens."""
        n_params = 50_000_000
        optimal = chinchilla_optimal_tokens(n_params)
        assert optimal == 1_000_000_000  # 50M * 20

    def test_chinchilla_1b_params(self):
        """1B params should give 20B tokens."""
        n_params = 1_000_000_000
        optimal = chinchilla_optimal_tokens(n_params)
        assert optimal == 20_000_000_000  # 1B * 20

    def test_chinchilla_custom_ratio(self):
        """Custom ratio should be applied."""
        n_params = 100_000_000
        optimal = chinchilla_optimal_tokens(n_params, ratio=10.0)
        assert optimal == 1_000_000_000  # 100M * 10

    def test_estimate_training_steps(self):
        """Training step estimation should be correct."""
        total_tokens = 1_000_000
        batch_size = 4
        chunk_size = 128
        gradient_accumulation = 2

        steps = estimate_training_steps(
            total_tokens=total_tokens,
            batch_size=batch_size,
            chunk_size=chunk_size,
            gradient_accumulation=gradient_accumulation,
        )

        # tokens_per_step = 4 * 128 * 2 = 1024
        # steps = 1_000_000 // 1024 = 976
        expected = total_tokens // (batch_size * chunk_size * gradient_accumulation)
        assert steps == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Test: DataLoader Integration
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDataLoaderIntegration:
    """Test PreTokenizedDataset with PyTorch DataLoader."""

    def test_dataloader_iteration(self, tiny_tokens_path):
        """Should be able to iterate through DataLoader."""
        from torch.utils.data import DataLoader

        dataset = PreTokenizedDataset(
            data_path=str(tiny_tokens_path),
            chunk_size=32,
            shard_length=128,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=tnt_collate_fn,
        )

        # Iterate through a few batches
        for i, batch in enumerate(dataloader):
            assert batch["input_ids"].shape == (4, 32) or batch["input_ids"].shape[0] <= 4
            assert batch["labels"].shape == batch["input_ids"].shape
            assert isinstance(batch["shard_boundary"], bool)
            if i >= 5:
                break

    def test_dataloader_no_shuffle_preserves_order(self, tiny_tokens_path):
        """Without shuffle, order should be preserved."""
        from torch.utils.data import DataLoader

        dataset = PreTokenizedDataset(
            data_path=str(tiny_tokens_path),
            chunk_size=32,
            shard_length=128,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=tnt_collate_fn,
        )

        batches = [batch for batch, _ in zip(dataloader, range(3))]

        # First batch should be first chunk
        raw_tokens = np.load(tiny_tokens_path)
        expected_first = torch.tensor(raw_tokens[0:32], dtype=torch.long)
        assert torch.equal(batches[0]["input_ids"].squeeze(), expected_first)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_chunk_size_larger_than_data(self, tmp_path):
        """Should handle case where chunk_size > data size."""
        # Create tiny dataset
        tiny = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        tiny_path = tmp_path / "tiny.npy"
        np.save(tiny_path, tiny)

        dataset = PreTokenizedDataset(
            data_path=str(tiny_path),
            chunk_size=10,  # Larger than data
            shard_length=20,
        )

        # Should have 0 chunks (not enough data)
        assert len(dataset) == 0

    def test_shard_length_equals_chunk_size(self, tiny_tokens_path):
        """When shard_length == chunk_size, every chunk after first is boundary."""
        chunk_size = 32
        dataset = PreTokenizedDataset(
            data_path=str(tiny_tokens_path),
            chunk_size=chunk_size,
            shard_length=chunk_size,  # Same as chunk
        )

        # Chunk 0: not boundary
        # Chunk 1: position 32 = shard_length, IS boundary
        # Chunk 2: position 64 = 2*shard_length, IS boundary
        assert dataset[0]["shard_boundary"] == False
        assert dataset[1]["shard_boundary"] == True
        assert dataset[2]["shard_boundary"] == True

    def test_token_dtype_preserved(self, tmp_path):
        """Token dtype should match what was saved."""
        # Create int32 tokens
        tokens = np.arange(1000, dtype=np.int32)
        path = tmp_path / "int32.npy"
        np.save(path, tokens)

        dataset = PreTokenizedDataset(str(path), chunk_size=32, shard_length=128)
        item = dataset[0]

        # Should be long tensor (int64) after conversion
        assert item["input_ids"].dtype == torch.long
