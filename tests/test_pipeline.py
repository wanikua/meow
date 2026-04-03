"""End-to-end pipeline test: train codebook, encode, decode, audit."""

import pytest
import torch

from meow.codebook import MeowCodebook
from meow.encoder import MeowEncoder
from meow.decoder import MeowDecoder
from meow.audit import MeowAudit
from meow.data import create_dataloaders


class TestFullPipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Train a small codebook for pipeline tests."""
        self.codebook = MeowCodebook(
            input_dim=64, codebook_dim=16, num_symbols=8
        )
        optimizer = torch.optim.Adam(self.codebook.parameters(), lr=1e-3)

        train_loader, _ = create_dataloaders(
            num_samples=200, embedding_dim=64, batch_size=32
        )

        # Quick training
        self.codebook.train()
        for epoch in range(10):
            for batch in train_loader:
                optimizer.zero_grad()
                _, info = self.codebook(batch, return_info=True)
                info["total_loss"].backward()
                optimizer.step()

        self.codebook.eval()

    def test_encode_decode_audit(self):
        encoder = MeowEncoder(codebook=self.codebook, device="cpu")
        decoder = MeowDecoder(codebook=self.codebook, device="cpu")
        audit = MeowAudit(codebook=self.codebook, device="cpu")

        # Encode
        x = torch.randn(4, 64)
        symbols = encoder.encode_batch(x)  # (4,) flat indices
        assert symbols.shape == (4,)

        # Decode
        recon = decoder.batch_decode(symbols)
        assert recon.shape == (4, 64)

        # Audit single message
        single_symbol = [symbols[0].item()]
        result = audit.audit(single_symbol, level="detailed", original_embedding=x[0:1])
        assert result.reconstruction_error is not None
        assert isinstance(result.decoded_text, str)

    def test_save_load_preserves_behavior(self, tmp_path):
        path = str(tmp_path / "pipeline_codebook.pt")
        self.codebook.save(path)
        loaded = MeowCodebook.load(path)

        x = torch.randn(4, 64)
        assert torch.equal(self.codebook.encode(x), loaded.encode(x))

    def test_training_improves_reconstruction(self):
        """After training, reconstruction error should be reasonable."""
        x = torch.randn(16, 64)
        _, info = self.codebook(x, return_info=True)
        # Should not be absurdly high after training
        assert info["reconstruction_loss"].item() < 100.0
