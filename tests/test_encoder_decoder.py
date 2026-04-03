"""Tests for MeowEncoder and MeowDecoder."""

import pytest
import torch
import numpy as np

from meow.codebook import MeowCodebook
from meow.encoder import MeowEncoder
from meow.decoder import MeowDecoder


@pytest.fixture
def codebook():
    return MeowCodebook(input_dim=128, codebook_dim=32, num_symbols=16)


@pytest.fixture
def encoder(codebook):
    return MeowEncoder(codebook=codebook, device="cpu")


@pytest.fixture
def decoder(codebook):
    return MeowDecoder(codebook=codebook, device="cpu")


class TestMeowEncoder:
    def test_encode_tensor(self, encoder):
        x = torch.randn(4, 128)
        symbols = encoder.encode(x, sequence_length=1)
        assert symbols.shape == (4,)

    def test_encode_tensor_multi_symbol(self, encoder):
        x = torch.randn(4, 128)
        symbols = encoder.encode(x, sequence_length=3)
        assert symbols.shape == (4, 3)

    def test_encode_single(self, encoder):
        x = torch.randn(128)
        symbols = encoder.encode(x, sequence_length=1)
        assert symbols.shape == (1,)

    def test_encode_numpy(self, encoder):
        x = np.random.randn(4, 128).astype(np.float32)
        symbols = encoder.encode(x, sequence_length=1)
        assert symbols.shape == (4,)

    def test_encode_list(self, encoder):
        x = [0.1] * 128
        symbols = encoder.encode(x, sequence_length=1)
        assert symbols.shape == (1,)

    def test_encode_batch(self, encoder):
        x = torch.randn(100, 128)
        symbols = encoder.encode_batch(x, batch_size=32)
        assert symbols.shape == (100,)

    def test_codebook_info(self, encoder):
        info = encoder.get_codebook_info()
        assert info["input_dim"] == 128
        assert info["num_symbols"] == 16

    def test_visualize_codebook(self, encoder):
        embeddings = encoder.visualize_codebook()
        assert embeddings.shape == (16, 32)


class TestMeowDecoder:
    def test_decode_tensor(self, decoder):
        symbols = torch.tensor([[5]])
        recon = decoder.decode(symbols)
        assert recon.shape == (1, 128)

    def test_decode_list(self, decoder):
        symbols = [5, 10, 3]
        recon = decoder.decode(symbols)
        assert recon.shape == (1, 128)

    def test_decode_to_text(self, decoder):
        symbols = torch.tensor([[5]])
        for level in ("summary", "medium", "detailed"):
            text = decoder.decode_to_text(symbols, level=level)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_decode_with_confidence(self, decoder):
        symbols = [5, 10]
        result = decoder.decode_with_confidence(symbols)
        assert "embedding" in result
        assert "confidence" in result

    def test_symbol_meaning(self, decoder):
        info = decoder.get_symbol_meaning(0)
        assert "embedding_norm" in info

    def test_batch_decode(self, decoder):
        symbols = torch.randint(0, 16, (50,))
        recon = decoder.batch_decode(symbols, batch_size=16)
        assert recon.shape == (50, 128)


class TestEncoderDecoderRoundtrip:
    def test_encode_decode_produces_valid_output(self, encoder, decoder):
        """Verify the full encode->decode pipeline runs without error."""
        x = torch.randn(8, 128)
        symbols = encoder.encode_batch(x)  # (8,) flat indices
        recon = decoder.batch_decode(symbols)  # (8, 128)
        assert recon.shape == x.shape

    def test_same_input_same_symbols(self, encoder):
        """Deterministic: same input produces same symbols."""
        x = torch.randn(4, 128)
        s1 = encoder.encode(x, sequence_length=1)
        s2 = encoder.encode(x, sequence_length=1)
        assert torch.equal(s1, s2)
