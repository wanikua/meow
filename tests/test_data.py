"""Tests for data module."""

import pytest
import torch
import numpy as np

from meow.data import SyntheticEmbeddingDataset, create_dataloaders


class TestSyntheticEmbeddingDataset:
    def test_shape(self):
        ds = SyntheticEmbeddingDataset(num_samples=100, embedding_dim=64)
        assert len(ds) == 100
        assert ds[0].shape == (64,)

    def test_deterministic(self):
        ds1 = SyntheticEmbeddingDataset(num_samples=50, embedding_dim=64, seed=42)
        ds2 = SyntheticEmbeddingDataset(num_samples=50, embedding_dim=64, seed=42)
        assert torch.equal(ds1[0], ds2[0])

    def test_different_seeds(self):
        ds1 = SyntheticEmbeddingDataset(num_samples=50, embedding_dim=64, seed=1)
        ds2 = SyntheticEmbeddingDataset(num_samples=50, embedding_dim=64, seed=2)
        assert not torch.equal(ds1[0], ds2[0])


class TestCreateDataloaders:
    def test_basic(self):
        train_loader, val_loader = create_dataloaders(
            num_samples=200, embedding_dim=64, batch_size=32
        )
        batch = next(iter(train_loader))
        assert batch.shape[1] == 64
        assert batch.shape[0] <= 32

    def test_val_split(self):
        train_loader, val_loader = create_dataloaders(
            num_samples=100, embedding_dim=64, batch_size=10, val_ratio=0.2
        )
        train_count = sum(len(b) for b in train_loader)
        val_count = sum(len(b) for b in val_loader)
        assert train_count == 80
        assert val_count == 20
