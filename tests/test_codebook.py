"""Tests for MeowCodebook and VectorQuantizer."""

import pytest
import torch

from meow.codebook import MeowCodebook, VectorQuantizer


@pytest.fixture
def vq():
    return VectorQuantizer(num_embeddings=16, embedding_dim=32)


@pytest.fixture
def codebook():
    return MeowCodebook(input_dim=128, codebook_dim=32, num_symbols=16)


class TestVectorQuantizer:
    def test_output_shapes(self, vq):
        x = torch.randn(8, 32)
        quantized, indices, info = vq(x)
        assert quantized.shape == (8, 32)
        assert indices.shape == (8,)
        assert all(k in info for k in ("loss", "perplexity", "usage_rate", "encoding_indices"))

    def test_indices_in_range(self, vq):
        x = torch.randn(64, 32)
        _, indices, _ = vq(x)
        assert indices.min() >= 0
        assert indices.max() < 16

    def test_gradient_passthrough(self, vq):
        x = torch.randn(8, 32, requires_grad=True)
        quantized, _, info = vq(x)
        info["loss"].backward()
        assert x.grad is not None

    def test_ema_updates_during_training(self, vq):
        vq.train()
        ema_before = vq.ema_cluster_size.clone()
        x = torch.randn(8, 32)
        vq(x)
        assert not torch.equal(ema_before, vq.ema_cluster_size)

    def test_no_ema_updates_during_eval(self, vq):
        vq.eval()
        ema_before = vq.ema_cluster_size.clone()
        x = torch.randn(8, 32)
        vq(x)
        assert torch.equal(ema_before, vq.ema_cluster_size)


class TestMeowCodebook:
    def test_forward_shapes(self, codebook):
        x = torch.randn(8, 128)
        recon, info = codebook(x, return_info=True)
        assert recon.shape == (8, 128)
        assert "total_loss" in info
        assert "reconstruction_loss" in info

    def test_encode_decode_shapes(self, codebook):
        x = torch.randn(8, 128)
        indices = codebook.encode(x)
        assert indices.shape == (8,)
        recon = codebook.decode(indices)
        assert recon.shape == (8, 128)

    def test_encode_indices_in_range(self, codebook):
        x = torch.randn(64, 128)
        indices = codebook.encode(x)
        assert indices.min() >= 0
        assert indices.max() < 16

    def test_training_reduces_loss(self, codebook):
        optimizer = torch.optim.Adam(codebook.parameters(), lr=1e-3)
        x = torch.randn(32, 128)

        # Initial reconstruction loss
        _, info0 = codebook(x, return_info=True)
        recon0 = info0["reconstruction_loss"].item()

        # Train a few steps
        for _ in range(50):
            optimizer.zero_grad()
            _, info = codebook(x, return_info=True)
            info["total_loss"].backward()
            optimizer.step()

        _, info_final = codebook(x, return_info=True)
        assert info_final["reconstruction_loss"].item() < recon0

    def test_save_load_roundtrip(self, codebook, tmp_path):
        path = str(tmp_path / "test_codebook.pt")
        codebook.save(path)
        loaded = MeowCodebook.load(path)

        assert loaded.input_dim == codebook.input_dim
        assert loaded.codebook_dim == codebook.codebook_dim
        assert loaded.num_symbols == codebook.num_symbols

        x = torch.randn(4, 128)
        idx_orig = codebook.encode(x)
        idx_loaded = loaded.encode(x)
        assert torch.equal(idx_orig, idx_loaded)
